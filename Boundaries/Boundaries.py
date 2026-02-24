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
        """
        Applica la condizione phi = V_value.
        A: matrice sparsa (es. scipy.sparse.lil_matrix)
        b: vettore dei termini noti
        """
        # Per le equazioni di Dirichlet, poniamo l'elemento diagonale a 1
        # e azzeriamo le connessioni con i vicini (questo andrà fatto durante l'assemblaggio).
        # E imponiamo b_k = V_value.
        
        # In una matrice lil_matrix possiamo assegnare in batch (più o meno)
        for k in self.k_indices:
            A.data[k] = [1.0]      # L'unico coefficiente sulla riga k sarà 1.0
            A.rows[k] = [k]        # Si troverà sulla colonna k (diagonale principale)
            b[k] = self.V_value

class NeumannBC3D(BoundaryCondition3D):
    def __init__(self, mask, mesh, direction):
        super().__init__(mask)
        self.direction = direction
        
        # Pre-calcoliamo gli indici dei nodi adiacenti (specchio) per tutti i nodi nella maschera
        # Usiamo unravel_index per passare dall'indice 1D (k) agli indici 3D (i, j, m)
        i, j, m = np.unravel_index(self.k_indices, shape=(mesh.N_x, mesh.N_y, mesh.N_z))
        
        # Modifichiamo l'indice in base alla direzione dello specchio
        if direction == 'z_up':      # Bordo a z=0, copiamo il valore da z=1
            m_adj = m + 1
            i_adj, j_adj = i, j
        elif direction == 'z_down':  # Bordo a z=Z_max, copiamo da z=Z_max-1
            m_adj = m - 1
            i_adj, j_adj = i, j
        # (Si possono aggiungere 'x_in', 'x_out' ecc. se servissero facce laterali di Neumann)
        else:
            raise ValueError(f"Direzione Neumann '{direction}' non gestita.")
            
        # Ricalcoliamo l'indice 1D del nodo adiacente
        self.k_adj_indices = np.ravel_multi_index((i_adj, j_adj, m_adj), dims=(mesh.N_x, mesh.N_y, mesh.N_z))

    def apply(self, A, b):
        """
        Applica la condizione phi_k - phi_adj = 0 (ovvero phi_k = phi_adj).
        """
        for k, k_adj in zip(self.k_indices, self.k_adj_indices):
            A.data[k] = [1.0, -1.0]     # Coefficienti dell'equazione
            A.rows[k] = [k, k_adj]      # Colonne: diagonale e nodo adiacente
            b[k] = 0.0