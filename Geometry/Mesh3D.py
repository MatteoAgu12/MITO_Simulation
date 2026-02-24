import numpy as np

class Mesh3D:
    def __init__(self, X_max, Y_max, Z_max, N_x, N_y, N_z):
        """
        Inizializza la griglia cartesiana 3D.
        Il dominio su X e Y è centrato in zero: [-X_max, X_max], [-Y_max, Y_max].
        Il dominio su Z parte dalla base dell'elettrodo: [0, Z_max].
        """
        self.N_x, self.N_y, self.N_z = N_x, N_y, N_z
        
        # Creazione degli assi
        self.x = np.linspace(-X_max, X_max, N_x)
        self.y = np.linspace(-Y_max, Y_max, N_y)
        self.z = np.linspace(0, Z_max, N_z)
        
        # Creazione della griglia 3D (X, Y, Z avranno forma N_x x N_y x N_z)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Mappa dei materiali
        self.material_map = np.zeros((N_x, N_y, N_z), dtype=int)

    def get_coordinates(self):
        return self.X, self.Y, self.Z