import cv2
import numpy as np

# Puntos 3D en el sistema de coordenadas del objeto (por ejemplo, las esquinas de un casco)
object_points = np.array([
    [0, 0, 0],    # Esquina 1
    [1, 0, 0],    # Esquina 2
    [1, 1, 0],    # Esquina 3
    [0, 1, 0]     # Esquina 4
], dtype=np.float32)

# Puntos 2D correspondientes en la imagen (estos puntos deben ser obtenidos a partir de la imagen)
image_points = np.array([
    [300, 200],   # Proyección de la Esquina 1
    [400, 200],   # Proyección de la Esquina 2
    [400, 300],   # Proyección de la Esquina 3
    [300, 300]    # Proyección de la Esquina 4
], dtype=np.float32)

# Matriz de cámara (parámetros intrínsecos)
camera_matrix = np.array([[1000, 0, 320], 
                          [0, 1000, 240], 
                          [0, 0, 1]], dtype=np.float32)

# Coeficientes de distorsión (asumimos una cámara sin distorsión para este ejemplo)
dist_coeffs = np.zeros((4,1))  # Si conoces los coeficientes de distorsión, colócalos aquí

# Resolver PnP usando EPnP
success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

if success:
    # Convertir el vector de rotación a una matriz de rotación
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    print("Matriz de rotación:")
    print(rotation_matrix)

    print("Vector de traslación:")
    print(translation_vector)
else:
    print("No se pudo calcular la pose.")
