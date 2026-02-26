import numpy as np
import matplotlib.pyplot as plt

# Importiamo i moduli
from Geometry.Mesh3D import Mesh3D
from Geometry.SystemGeometry import SystemGeometry
from Boundaries.SystemBoundaries import SystemBoundaries3D
from Materials.Materials import Material
from Solver.FVSolver import FVSolver3D
from GUI.plot_slice import plot_slice_yz
from PostProcess.Impedance import calculate_impedance

def main():
    print("Inizializzazione del sistema...")
    
    # 1. Setup dei parametri geometrici
    X_max, Y_max, Z_max = 203.0, 203.0, 260
    N_x, N_y, N_z = 30, 30, 50
    h1_val, h_val = 20, 230.0
    a_e = 70.0  # Raggio elettrodo
    V_0 = 10.0   # Potenziale all'elettrodo in Volt
    f_sim = 1e5 # Frequenza di simulazione in Hz
    
    # 2. Creazione della Mesh e Geometria
    print("Generazione Mesh e Geometria...")
    mesh = Mesh3D(X_max, Y_max, Z_max, N_x, N_y, N_z)
    sys_geo = SystemGeometry(mesh)
    sys_geo.build_system(
        h1=h1_val, h=h_val, 
        r_aperture_bottom=5.0, r_aperture_top=200.0,
        cell_r=0.0 # Nessuna cellula
    )
    
    # 3. Definizione dei Materiali
    print("Assegnazione Proprietà Fisiche...")
    # Usiamo valori tipici: Idrogelo leggermente meno conduttivo dell'Elettrolita
    materials_dict = {
        0: Material("Hydrogel", sigma=0.5, epsilon_r=60),
        1: Material("Electrolyte", sigma=1.5, epsilon_r=78)
    }
    
    # 4. Creazione delle Condizioni al Contorno
    print("Inizializzazione Condizioni al Contorno...")
    sys_bounds = SystemBoundaries3D(mesh, a_e, V_0)
    
    frequencies = np.logspace(1, 5, 10)
    Z = []
    phase = []
    for f in frequencies:
        # =========================================================
        # 5. ESECUZIONE DEL SOLUTORE FVM
        # =========================================================
        print(f"Inizializzazione Solutore per f = {f} Hz...")
        solver = FVSolver3D(mesh, materials_dict)

        # Calcoliamo la vera distribuzione del potenziale complesso phi
        phi_complex = solver.assemble_and_solve(f=f, boundaries_manager=sys_bounds)

        # Estraiamo l'ampiezza (modulo) del potenziale per la visualizzazione
        phi_amplitude = np.abs(phi_complex)

        # =========================================================
        # 6. POST-PROCESSING: CALCOLO CORRENTE E IMPEDENZA
        # =========================================================
        print("Estrazione della corrente all'elettrodo...")
        Z_tot, I_tot = calculate_impedance(mesh, phi_complex, V_0, a_e, f, materials_dict)

        # L'impedenza è un numero complesso. Ne calcoliamo Modulo e Fase
        Z_mag = np.abs(Z_tot)
        # np.angle restituisce la fase in radianti, la convertiamo in gradi
        Z_phase = np.degrees(np.angle(Z_tot)) 

        Z.append(Z_mag)
        phase.append(Z_phase)

        print("\n" + "="*40)
        print(" RISULTATI DELLA SIMULAZIONE EIS")
        print("="*40)
        print(f" Frequenza        : {f} Hz")
        print(f" Corrente Totale  : {np.abs(I_tot):.4e} A")
        print(f" Modulo Impedenza : {Z_mag:.2f} Ohm")
        print(f" Fase Impedenza   : {Z_phase:.2f}°")
        print("="*40 + "\n")
    
    # =========================================================
    # 7. VISUALIZZAZIONE DEI RISULTATI
    # =========================================================
    print("Apertura del grafico 2D con la soluzione reale...")
    phi_amplitude = np.abs(phi_complex)
    plot_slice_yz(mesh, phi_amplitude, a_e)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(frequencies, Z, label='Z')
    ax[1].plot(frequencies, phase, label=r'$\phi$')
    plt.show()

if __name__ == "__main__":
    main()