from Geometry.Regions import PiramidalFrustumAperture, SphericalCell

class SystemGeometry:
    def __init__(self, mesh):
        self.mesh = mesh
        self.X, self.Y, self.Z = mesh.get_coordinates()
        
    def build_system(self, h1, h, r_aperture_bottom, r_aperture_top, 
                     cell_x=0.0, cell_y=0.0, cell_z=70.0, cell_r=0.0):
        # 0 = Hydrogel, 1 = Electrolyte, 2 = Cell
        
        # Sfondo: Elettrolita
        self.mesh.material_map.fill(1)
        
        # Strato di Idrogelo alla base
        hydrogel_mask = self.Z <= (h1 + h)
        self.mesh.material_map[hydrogel_mask] = 0
        
        # Scavo del pozzetto (riempito di Elettrolita)
        aperture = PiramidalFrustumAperture(z_bottom=h1, z_top=h1+h, 
                                     r_bottom=r_aperture_bottom, r_top=r_aperture_top)
        self.mesh.material_map[aperture.contains(self.X, self.Y, self.Z)] = 1
        
        # Inserimento della cellula (ora supporta coordinate x, y arbitrarie)
        if cell_r > 0:
            cell = SphericalCell(x_c=cell_x, y_c=cell_y, z_c=cell_z, radius=cell_r)
            self.mesh.material_map[cell.contains(self.X, self.Y, self.Z)] = 2