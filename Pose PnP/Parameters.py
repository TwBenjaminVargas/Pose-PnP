# Mantiene diccionarios de parametros de configuracion para el sistema

#librerias
import numpy as np

class Parameters:
    
    # Constructor
    def __init__(self,
                 # Parametros intrinsecos de la camara
                 f_x = 1000,
                 f_y = 1000,
                 x = 320,
                 y = 240,
                 # Coeficientes de distorsión
                dist_coeffs = np.zeros((4, 1))
                ):
        
        #PARAMETROS DE CAMARA
        
        self.cameraParams = dict()
        
        # Matriz intrinseca
        self.cameraParams["K"] = np.array([
                            [f_x, 0, x],
                            [0, f_y, y],
                            [0, 0, 1]
                            ], dtype=np.float32)
        
        # Coeficientes de distorsión
        self.cameraParams["dist_coeffs"] = dist_coeffs
        
        # PARAMETROS DE FILTRADO
        
        
    