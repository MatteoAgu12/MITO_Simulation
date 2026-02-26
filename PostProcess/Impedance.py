import numpy as np

def calculate_impedance(mesh, phi, V_0, a_e, f, materials_dict):
    """
    Calcola la corrente totale e l'impedenza del biosensore.
    
    Parametri:
    - mesh: oggetto Mesh3D
    - phi: array 3D del potenziale complesso risolto
    - V_0: tensione applicata all'elettrodo [V]
    - a_e: raggio dell'elettrodo [µm]
    - f: frequenza di simulazione [Hz]
    - materials_dict: dizionario dei materiali per ricalcolare Sigma
    
    Ritorna:
    - Z: Impedenza complessa [Ohm]
    - I_tot: Corrente complessa totale [A]
    """
    # 1. Passi spaziali della griglia (convertiti da µm a metri se i materiali usano S/m)
    # Assumiamo che dx, dy, dz calcolati dalla mesh siano in micrometri.
    # NOTA: Per avere l'impedenza in Ohm (reali), le distanze devono essere in metri!
    # Convertiamo i passi in metri moltiplicando per 1e-6.
    dx_m = (mesh.x[1] - mesh.x[0]) * 1e-6
    dy_m = (mesh.y[1] - mesh.y[0]) * 1e-6
    dz_m = (mesh.z[1] - mesh.z[0]) * 1e-6
    a_e_m = a_e * 1e-6
    
    # 2. Troviamo i nodi che si trovano fisicamente "sopra" l'elettrodo
    # L'elettrodo è a z=0 (indice k=0). Prendiamo la fetta XY.
    X_m = mesh.X[:, :, 0] * 1e-6
    Y_m = mesh.Y[:, :, 0] * 1e-6
    R_m = np.sqrt(X_m**2 + Y_m**2)
    
    # Maschera booleana: True solo per i voxel dentro il raggio a_e
    electrode_mask = (R_m <= a_e_m)
    
    # 3. Ricostruiamo la conduttività (Sigma) per lo strato a contatto con l'elettrodo
    # Prendiamo la mappa dei materiali al primo livello Z utile (indice k=1)
    mat_map_slice = mesh.material_map[:, :, 1]
    Sigma_slice = np.zeros_like(mat_map_slice, dtype=complex)
    
    for mat_id, material in materials_dict.items():
        Sigma_slice[mat_map_slice == mat_id] = material.get_complex_conductivity(f)
        
    # 4. Calcolo della caduta di potenziale tra elettrodo (k=0) e primo nodo (k=1)
    # phi a k=0 vale esattamente V_0 (imposto dalle BC).
    # phi a k=1 è il potenziale appena calcolato dal solutore.
    dV = V_0 - phi[:, :, 1]
    
    # 5. Calcolo della corrente locale J * Area
    # I = Sigma * (dV / dz) * dx * dy
    I_local = Sigma_slice * (dV / dz_m) * (dx_m * dy_m)
    
    # 6. Somma della corrente SOLO sull'area dell'elettrodo
    I_tot = np.sum(I_local[electrode_mask])
    
    # 7. Calcolo Impedenza: Z = V / I
    Z = V_0 / I_tot
    
    return Z, I_tot