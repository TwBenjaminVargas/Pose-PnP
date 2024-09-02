import cv2
import numpy as np

# Leer imagen
image = cv2.imread('/home/ben/Documentos/Pose PnP/OpenCV EPnP/Examples/A4.jpg')

# Verificar si la imagen fue cargada correctamente
if image is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar esquinas usando Shi-Tomasi
corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=10)
corners = np.intp(corners)  # Actualizar para usar `np.intp` en lugar de `np.int0`

# Verificar si se detectaron suficientes esquinas
if corners is None or len(corners) < 4:
    print("No se detectaron suficientes esquinas.")
    exit()

# Dibujar las esquinas detectadas
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Mostrar la imagen con los puntos detectados
cv2.imshow('Detected Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Definir los puntos 3D en el sistema de coordenadas del objeto (esquinas de la hoja de papel)
object_points = np.array([
    [0, 0, 0],       # Esquina inferior izquierda
    [210, 0, 0],     # Esquina inferior derecha
    [210, 297, 0],   # Esquina superior derecha
    [0, 297, 0]      # Esquina superior izquierda
], dtype=np.float32)

# Definir los puntos 2D en la imagen (debe corresponder con los puntos detectados)
image_points = np.array([
    [corners[0][0][0], corners[0][0][1]],
    [corners[1][0][0], corners[1][0][1]],
    [corners[2][0][0], corners[2][0][1]],
    [corners[3][0][0], corners[3][0][1]]
], dtype=np.float32)

# Matriz de cámara (parámetros intrínsecos)
camera_matrix = np.array([[1000, 0, 320], 
                          [0, 1000, 240], 
                          [0, 0, 1]], dtype=np.float32)

# Coeficientes de distorsión (asumimos una cámara sin distorsión para este ejemplo)
dist_coeffs = np.zeros((4,1))

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
