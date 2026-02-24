import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from Geometry.Mesh3D import Mesh3D
from Geometry.SystemGeometry import SystemGeometry

if __name__ == "__main__":
    # Parametri geometrici
    X_max, Y_max, Z_max = 80.0, 80.0, 150.0
    # Usiamo 81 in modo che l'indice centrale sia esattamente lo zero
    N_x, N_y, N_z = 81, 81, 150 
    
    mesh = Mesh3D(X_max, Y_max, Z_max, N_x, N_y, N_z)
    sys_geo = SystemGeometry(mesh)
    
    # Inseriamo la cellula decentrata lungo l'asse Y
    sys_geo.build_system(
        h1=40.0, h=60.0, 
        r_aperture_bottom=20.0, r_aperture_top=45.0,
        cell_x=0.0, cell_y=15.0, cell_z=70.0, cell_r=15.0
    )
    
    # Trova l'indice esatto corrispondente a x = 0
    x_mid_idx = N_x // 2
    print(f"Slice estratto a x = {mesh.x[x_mid_idx]} µm")
    
    # Estraiamo la fetta YZ dalla matrice 3D
    slice_yz = mesh.material_map[x_mid_idx, :, :]
    
    # Setup dei colori: 0=Hydrogel(blu), 1=Electrolyte(grigio), 2=Cell(verde)
    cmap = ListedColormap(['#a0c4ff', '#f8f9fa', '#bbf7d0'])
    
    plt.figure(figsize=(7, 8))
    
    # Plottiamo la fetta (trasponiamo con .T perché pcolormesh usa Y, Z)
    plt.pcolormesh(mesh.y, mesh.z, slice_yz.T, cmap=cmap, shading='auto')
    
    # Aggiungiamo l'elettrodo sul fondo per chiarezza (da -a_e a +a_e lungo Y)
    a_e = 40.0 
    plt.plot([-a_e, a_e], [0, 0], color='gray', linewidth=5, label='Electrode')
    
    # Grafica
    plt.title(r"Slice 2D sul piano YZ (a $x=0$)")
    plt.xlabel(r"Asse Y ($\mu m$)")
    plt.ylabel(r"Asse Z ($\mu m$)")
    
    legend_patches = [
        mpatches.Patch(color='#a0c4ff', label='Hydrogel'),
        mpatches.Patch(color='#f8f9fa', label='Electrolyte'),
        mpatches.Patch(color='#bbf7d0', label='Cell'),
        mpatches.Patch(color='gray', label='Electrode')
    ]
    plt.legend(handles=legend_patches, loc='upper right')
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()