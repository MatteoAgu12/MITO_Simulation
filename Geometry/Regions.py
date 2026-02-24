import numpy as np

class Region3D:
    def contains(self, X, Y, Z):
        raise NotImplementedError()

class SphericalCell(Region3D):
    def __init__(self, x_c, y_c, z_c, radius):
        """Ora la cellula può essere posizionata ovunque nello spazio (x_c, y_c)."""
        self.x_c = x_c
        self.y_c = y_c
        self.z_c = z_c
        self.r_s = radius
        
    def contains(self, X, Y, Z):
        # Equazione della sfera 3D
        return ((X - self.x_c)**2 + (Y - self.y_c)**2 + (Z - self.z_c)**2) <= self.r_s**2

class PiramidalFrustumAperture(Region3D):
    def __init__(self, z_bottom, z_top, r_bottom, r_top):
        """Il pozzetto rimane simmetrico rispetto all'asse Z centrale (x=0, y=0)."""
        self.z_b = z_bottom
        self.z_t = z_top
        self.r_b = r_bottom
        self.r_t = r_top
        
    def contains(self, X, Y, Z):
        if self.z_t == self.z_b:
            return np.zeros_like(X, dtype=bool)
            
        # Filtro sull'asse Z
        z_mask = (Z >= self.z_b) & (Z <= self.z_t)
        
        # Raggio del tronco di cono in funzione di Z
        slope = (self.r_t - self.r_b) / (self.z_t - self.z_b)
        R_z = self.r_b + slope * (Z - self.z_b)
        
        # Distanza radiale dal centro (x=0, y=0)
        R_xy_squared = X**2 + Y**2
        
        return z_mask & (R_xy_squared <= R_z**2)