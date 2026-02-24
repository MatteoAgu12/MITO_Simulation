import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import matplotlib.patches as mpatches

from Geometry.Mesh3D import Mesh3D
from Geometry.SystemGeometry import SystemGeometry

if __name__ == "__main__":
    # Parametri geometrici
    X_max, Y_max, Z_max = 203.0, 203.0, 400.0
    N_x, N_y, N_z = 80, 80, 100
    h1_val, h_val = 100.0, 230.0
    a_e = 70.0  # Raggio dell'elettrodo in micrometri (uguale a X_max in questo esempio)
    
    mesh = Mesh3D(X_max, Y_max, Z_max, N_x, N_y, N_z)
    sys_geo = SystemGeometry(mesh)
    
    sys_geo.build_system(
        h1=h1_val, h=h_val, 
        r_aperture_bottom=5.0, r_aperture_top=200.0,
        cell_x=15.0, cell_y=0.0, cell_z=70.0, cell_r=0
    )
    
    # 1. Creiamo volumi float (0.0 = Hydrogel, 1.0 = Altro)
    vol_data = mesh.material_map.copy()
    vol_data[vol_data > 0] = 1
    hydrogel_vol = (vol_data == 0).astype(float)
    
    dx = mesh.x[1] - mesh.x[0]
    dy = mesh.y[1] - mesh.y[0]
    dz = mesh.z[1] - mesh.z[0]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 2. Plottiamo l'isosuperficie dell'Idrogel (Trasparente)
    verts_h, faces_h, normals_h, values_h = marching_cubes(hydrogel_vol, level=0.5, spacing=(dx, dy, dz))
    verts_h[:, 0] -= X_max; verts_h[:, 1] -= Y_max # Centriamo
    
    mesh_hydrogel = Poly3DCollection(verts_h[faces_h], alpha=0.2, facecolor='#a0c4ff', edgecolor='none')
    ax.add_collection3d(mesh_hydrogel)
    
    # Creiamo una griglia polare per il disco dell'elettrodo
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, a_e, 50)
    T, R = np.meshgrid(theta, r)
    
    # Convertiamo in coordinate cartesiane
    X_elec = R * np.cos(T)
    Y_elec = R * np.sin(T)
    Z_elec = np.zeros_like(X_elec)  # L'elettrodo è posizionato a z=0
    
    # Disegniamo la superficie dorata
    ax.plot_surface(X_elec, Y_elec, Z_elec, color='gold', alpha=0.7, edgecolor='none')

    # --- MODIFICA 2: AGGIUNTA CONTORNI NERI A DIFFERENTI Z ---
    z_levels_um = [0, h1_val, h1_val + h_val/2, h1_val + h_val, Z_max]
    
    for z_target in z_levels_um:
        z_idx = np.argmin(np.abs(mesh.z - z_target))
        actual_z = mesh.z[z_idx]
        slice_mat = vol_data[:, :, z_idx]
        
        if np.any(slice_mat == 0) and np.any(slice_mat == 1):
            ax.contour(mesh.X[:,:,z_idx], mesh.Y[:,:,z_idx], slice_mat, 
                       levels=[0.5], zdir='z', offset=actual_z, colors='black', linewidths=2.0)

    # 4. Impostazioni finali
    ax.set_xlim(-X_max, X_max); ax.set_ylim(-Y_max, Y_max); ax.set_zlim(0, Z_max)
    ax.set_xlabel(r"X ($\mu m$)"); ax.set_ylabel(r"Y ($\mu m$)"); ax.set_zlabel(r"Z ($\mu m$)")
    ax.set_title("Sistema 3D Completo con Contorni e Elettrodo d'Oro")
    
    # Trucco per aggiungere l'elettrodo alla legenda
    legend_patches = [
        mpatches.Patch(color='#a0c4ff', label='Hydrogel', alpha=0.5),
        mpatches.Patch(color='gold', label='Gold Electrode', alpha=0.7)
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    
    # Vista dall'alto per apprezzare i cerchi concentrici
    ax.view_init(elev=35, azim=-45)
    plt.tight_layout()
    plt.show()