import numpy as np

class Material:
    def __init__(self, name, sigma, epsilon_r):
        """
        Definisce un materiale di volume (idrogelo, elettrolita, citoplasma).
        
        Parametri:
        - name: Nome del materiale (stringa)
        - sigma: Conducibilità elettrica in [S/m]
        - epsilon_r: Permettività relativa (adimensionale). 
                     Verrà moltiplicata per epsilon_0 internamente.
        """
        self.name = name
        self.sigma = sigma
        self.epsilon_0 = 8.854e-12 # Permettività del vuoto [F/m]
        self.epsilon = epsilon_r * self.epsilon_0
        
    def get_complex_conductivity(self, f):
        """
        Calcola la conduttività complessa Sigma = sigma + j * 2*pi*f * epsilon
        per una data frequenza f [Hz].
        """
        omega = 2 * np.pi * f
        return self.sigma + 1j * omega * self.epsilon

class CellMembrane:
    def __init__(self, G_m, C_m):
        """
        Definisce le proprietà di interfaccia della membrana cellulare.
        
        Parametri:
        - G_m: Conduttanza specifica della membrana in [S/m^2]
        - C_m: Capacità specifica della membrana in [F/m^2]
        """
        self.G_m = G_m
        self.C_m = C_m
        
    def get_admittance(self, f):
        """
        Calcola l'ammettenza complessa Y_m = G_m + j * 2*pi*f * C_m
        per una data frequenza f [Hz].
        """
        omega = 2 * np.pi * f
        return self.G_m + 1j * omega * self.C_m
    
class PEDOT:
    def __init__(self, G_m, C_m):
        """
        Definisce le proprietà di interfaccia di elettrodo in PEDOT:PSS.
        
        Parametri:
        - G_m: conductance of the file [S/m^2]
        - C_m: capacità volumetrica PEDOT [F/m^2]
        """
        self.G_m = G_m
        self.C_m = C_m
        
    def get_admittance(self, f):
        """
        Calcola l'ammettenza complessa Y_m = G_m + j * 2*pi*f * C_m
        per una data frequenza f [Hz].
        """
        omega = 2 * np.pi * f
        return self.G_m + 1j * omega * self.C_m