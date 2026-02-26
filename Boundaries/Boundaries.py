import numpy as np

class BoundaryCondition3D:
    def __init__(self, mask):
        """
        mask: Un array booleano 3D (N_x, N_y, N_z). 
              I nodi in cui mask == True sono quelli a cui applicare la BC.
        """
        self.mask = mask
        # np.flatnonzero restituisce gli indici 1D (per la matrice sparsa) dove mask è True
        self.k_indices = np.flatnonzero(mask)

class DirichletBC3D(BoundaryCondition3D):
    def __init__(self, mask, V_value):
        super().__init__(mask)
        self.V_value = V_value

    def apply(self, A, b):
        k = self.k_indices
        for kk in k:               # LIL non supporta bene batch completo, ma riduciamo overhead
            A.rows[kk] = [int(kk)]
            A.data[kk] = [1.0 + 0.0j]
        b[k] = self.V_value


class NeumannBC3D(BoundaryCondition3D):
    def __init__(self, mask, mesh, direction):
        super().__init__(mask)
        self.direction = direction

        i, j, m = np.unravel_index(self.k_indices, shape=(mesh.N_x, mesh.N_y, mesh.N_z))
        if direction == 'z_up':
            i_adj, j_adj, m_adj = i, j, m + 1
        elif direction == 'z_down':
            i_adj, j_adj, m_adj = i, j, m - 1
        else:
            raise ValueError(f"Direzione Neumann '{direction}' non gestita.")

        self.k_adj_indices = np.ravel_multi_index(
            (i_adj, j_adj, m_adj), dims=(mesh.N_x, mesh.N_y, mesh.N_z)
        )

    def apply(self, A, b):
        for k, k_adj in zip(self.k_indices, self.k_adj_indices):
            A.rows[k] = [int(k), int(k_adj)]
            A.data[k] = [1.0 + 0.0j, -1.0 + 0.0j]
        b[self.k_indices] = 0.0
        