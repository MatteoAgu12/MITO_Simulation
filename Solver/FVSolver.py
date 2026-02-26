import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class FVSolver3D:
    def __init__(self, mesh, materials_dict):
        self.mesh = mesh
        self.materials_dict = materials_dict
        self.N_nodes = mesh.N_x * mesh.N_y * mesh.N_z

        self.dx = mesh.x[1] - mesh.x[0]
        self.dy = mesh.y[1] - mesh.y[0]
        self.dz = mesh.z[1] - mesh.z[0]

        # cache indici lineari 3D -> 1D
        self._K = np.arange(self.N_nodes, dtype=np.int64).reshape(
            mesh.N_x, mesh.N_y, mesh.N_z
        )

    def _build_sigma_map(self, f):
        Sigma = np.zeros_like(self.mesh.material_map, dtype=np.complex128)
        for mat_id, material in self.materials_dict.items():
            Sigma[self.mesh.material_map == mat_id] = material.get_complex_conductivity(f)
        return Sigma

    @staticmethod
    def _harmonic_mean(S1, S2):
        eps = 1e-15
        return (2.0 * S1 * S2) / (S1 + S2 + eps)

    def _assemble_matrix_internal_faces(self, Sigma):
        Nx, Ny, Nz = self.mesh.N_x, self.mesh.N_y, self.mesh.N_z
        K = self._K

        rows_parts = []
        cols_parts = []
        data_parts = []

        def add_face_couplings(P, Q, G):
            # P, Q, G sono array 3D/ND allineati (flattenabili)
            p = P.ravel()
            q = Q.ravel()
            g = G.ravel().astype(np.complex128, copy=False)

            rows_parts.extend([p, q, p, q])
            cols_parts.extend([p, q, q, p])
            data_parts.extend([g, g, -g, -g])

        # Facce interne X (i <-> i+1)
        S1 = Sigma[:-1, :, :]
        S2 = Sigma[1:, :, :]
        Gx = self._harmonic_mean(S1, S2) * (self.dy * self.dz / self.dx)
        add_face_couplings(K[:-1, :, :], K[1:, :, :], Gx)

        # Facce interne Y (j <-> j+1)
        S1 = Sigma[:, :-1, :]
        S2 = Sigma[:, 1:, :]
        Gy = self._harmonic_mean(S1, S2) * (self.dx * self.dz / self.dy)
        add_face_couplings(K[:, :-1, :], K[:, 1:, :], Gy)

        # Facce interne Z (k <-> k+1)
        S1 = Sigma[:, :, :-1]
        S2 = Sigma[:, :, 1:]
        Gz = self._harmonic_mean(S1, S2) * (self.dx * self.dy / self.dz)
        add_face_couplings(K[:, :, :-1], K[:, :, 1:], Gz)

        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts)

        A = sp.coo_matrix((data, (rows, cols)),
                          shape=(self.N_nodes, self.N_nodes),
                          dtype=np.complex128)
        A.sum_duplicates()
        return A

    def assemble_and_solve(self, f, boundaries_manager):
        print(f"--- Inizio Soluzione per f = {f} Hz ---")

        print("  -> Mappa conduttività complessa...")
        Sigma = self._build_sigma_map(f)

        print("  -> Assemblaggio matrice (facce interne, no wrap-around)...")
        A = self._assemble_matrix_internal_faces(Sigma)

        b = np.zeros(self.N_nodes, dtype=np.complex128)

        print("  -> Applicazione Boundary Conditions...")
        A_lil = A.tolil()
        boundaries_manager.apply_all(A_lil, b)

        # Niente regolarizzazione di default (se serve, riattivala solo in casi patologici)
        # A_lil.setdiag(A_lil.diagonal() + 1e-12)

        A_csr = A_lil.tocsr()
        A_csr.sum_duplicates()
        A_csr.sort_indices()

        # Precondizionatore ILU (opzionale)
        M = None
        try:
            print("  -> Costruzione precondizionatore ILU...")
            ilu = spla.spilu(A_csr.tocsc(), drop_tol=1e-3, fill_factor=8)
            M = spla.LinearOperator(A_csr.shape, matvec=ilu.solve, dtype=np.complex128)
            print("     ILU ok")
        except Exception as e:
            print(f"     ILU fallito ({e}), continuo senza precondizionatore")

        print("  -> Risoluzione del sistema (BiCGSTAB)...")
        it = {"n": 0}
        def callback(_):
            it["n"] += 1
            if it["n"] % 25 == 0:
                print(f"     iter {it['n']}")

        phi_1d, info = spla.bicgstab(
            A_csr, b,
            rtol=1e-8, atol=1e-12,
            maxiter=2000,
            M=M,
            callback=callback
        )

        if info != 0:
            print(f"  -> BiCGSTAB info={info}; fallback a GMRES...")
            phi_1d, info = spla.gmres(
                A_csr, b,
                restart=50,
                rtol=1e-8, atol=1e-12,
                maxiter=500,
                M=M
            )
            if info != 0:
                raise RuntimeError(f"GMRES non convergente (info={info})")

        phi_3d = phi_1d.reshape((self.mesh.N_x, self.mesh.N_y, self.mesh.N_z))

        # sanity check veloce
        if not np.all(np.isfinite(phi_3d)):
            raise RuntimeError("La soluzione contiene NaN/Inf")

        print("  -> Soluzione completata")
        return phi_3d