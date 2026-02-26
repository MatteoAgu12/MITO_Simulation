import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_slice_yz(mesh, phi, a_e):
    """
    Estrae e plotta una sezione YZ (passante per X=0) del sistema.
    
    - mesh: l'oggetto Mesh3D contenente le coordinate e la material_map
    - phi: array 3D del potenziale da plottare
    - a_e: raggio dell'elettrodo (per disegnarlo sul fondo)
    """
    # Troviamo l'indice centrale sull'asse X (corrisponde a x=0 se N_x è dispari)
    x_mid_idx = mesh.N_x // 2
    
    # Estraiamo le matrici 2D corrispondenti a quella fetta
    slice_y = mesh.Y[x_mid_idx, :, :]
    slice_z = mesh.Z[x_mid_idx, :, :]
    slice_pot = phi[x_mid_idx, :, :]
    slice_mat = mesh.material_map[x_mid_idx, :, :]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # A. Mappa dei colori per il potenziale
    # vmin e vmax fissano i limiti dei colori. Preveniamo errori se max è 0.
    v_max = np.max(np.abs(phi))
    if v_max == 0: v_max = 1.0 
    
    cmap_plot = ax.pcolormesh(slice_y, slice_z, slice_pot, shading='auto', 
                              cmap='magma', vmin=0, vmax=v_max)
    plt.colorbar(cmap_plot, ax=ax, label='Potenziale Elettrico (V)')
    
    # B. Aggiunta dei contorni dei materiali (Gel vs Elettrolita)
    if np.any(slice_mat == 0) and np.any(slice_mat == 1):
        ax.contour(slice_y, slice_z, slice_mat, levels=[0.5], 
                   colors='white', linewidths=2.5, linestyles='dashed')
        
    # C. Aggiunta dell'Elettrodo d'Oro (linea da -a_e a +a_e a z=0)
    # ax.plot([-a_e, a_e], [0, 0], color='gold', linewidth=6)
    
    # D. Estetica e legende
    ax.set_title(f"Distribuzione del Potenziale (Slice YZ a x = {mesh.x[x_mid_idx]:.1f} µm)")
    ax.set_xlabel(r"Asse Y ($\mu m$)")
    ax.set_ylabel(r"Asse Z ($\mu m$)")
    
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='white', linestyle='--', linewidth=2.5, label='Bordo Idrogelo/Elettrolita'),
        mpatches.Patch(facecolor='gold', label='Elettrodo in Oro')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.axis('equal') # Proporzioni corrette
    plt.tight_layout()
    plt.show()