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
if corners is not None:
    corners = np.int0(corners)  # Convertir a enteros
else:
    print("No se detectaron suficientes esquinas.")
    exit()

# Dibujar las esquinas detectadas
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

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
    
    # Definir los ejes del sistema de coordenadas del objeto
    axis = np.float32([[50,0,0], [0,50,0], [0,0,-50]]).reshape(-1, 3)
    
    # Proyectar los puntos 3D del eje en la imagen 2D
    imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    # Convertir los puntos proyectados a enteros
    imgpts = np.int0(imgpts)
    
    # Dibujar los ejes en la imagen
    corner = tuple(np.int0(image_points[0]))  # Convertir a tupla de enteros
    image = cv2.line(image, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)  # Eje X (Rojo)
    image = cv2.line(image, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Eje Y (Verde)
    image = cv2.line(image, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)  # Eje Z (Azul)
    
    # Mostrar la imagen con los ejes proyectados
    cv2.imshow('Pose Estimation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Matriz de rotación:")
    print(rotation_matrix)

    print("Vector de traslación:")
    print(translation_vector)
else:
    print("No se pudo calcular la pose.")
