import numpy as np

from Boundaries.Boundaries import DirichletBC3D, NeumannBC3D

class SystemBoundaries3D:
    def __init__(self, mesh, a_e, V_0):
        self.mesh = mesh
        self.a_e = a_e
        self.V_0 = V_0
        self.bcs = []
        self._build_boundaries()
        
    def _build_boundaries(self):
        X, Y, Z = self.mesh.get_coordinates()
        
        # Tolleranze per i confronti con float
        tol = 1e-6
        X_max = np.max(X)
        Y_max = np.max(Y)
        Z_max = np.max(Z)
        
        # 1. Elettrodo (Dirichlet V=V_0)
        # Sul fondo (Z == 0) e all'interno del raggio a_e
        electrode_mask = (Z <= tol) & ((X**2 + Y**2) <= self.a_e**2 + tol)
        self.bcs.append(DirichletBC3D(electrode_mask, self.V_0))
        
        # 2. Substrato Isolante (Neumann dPhi/dz = 0)
        # Sul fondo (Z == 0) e all'esterno del raggio a_e
        substrate_mask = (Z <= tol) & ((X**2 + Y**2) > self.a_e**2 + tol)
        self.bcs.append(NeumannBC3D(substrate_mask, self.mesh, direction='z_up'))
        
        # 3. Cima (Dirichlet V=0, Far-Field)
        top_mask = (Z >= Z_max - tol)
        self.bcs.append(DirichletBC3D(top_mask, 0.0))
        
        # 4. Facce Laterali (Dirichlet V=0, Far-Field)
        side_mask = (np.abs(X - X_max) <= tol) | (np.abs(X + X_max) <= tol) | \
                    (np.abs(Y - Y_max) <= tol) | (np.abs(Y + Y_max) <= tol)
        self.bcs.append(DirichletBC3D(side_mask, 0.0))

    def apply_all(self, A, b):
        """Applica in sequenza tutte le condizioni al contorno sulla matrice A e vettore b"""
        for bc in self.bcs:
            bc.apply(A, b)