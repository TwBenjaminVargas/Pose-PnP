import cv2
import numpy as np

# Load image
image = cv2.imread('blacksquare.jpg')
#Square Angles coordenades: (Upper left to Bottom left)
#A1: (30,28)
#A2: (474,28)
#A3: (474,474)
#A4: (30,474)
if image is None:
    print("No se encontro imagen")
    exit(0)
# Show image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2D image coordenades
image_points = np.array([
    [30, 28],     # A1
    [474, 28],    # A2
    [474, 474],   # A3
    [30, 474]     # A4
], dtype=np.float32)

# 3D object coordenades
object_points = np.array([
    [0, 0, 0],    # A1
    [1, 0, 0],    # A2
    [1, 1, 0],    # A3
    [0, 1, 0]     # A4
], dtype=np.float32)


#calibracion de la camara
# Parámetros intrínsecos de la cámara (valores aproximados)
f_x = 1000
f_y = 1000
cx = 320
cy = 240

# Matriz intrínseca
K = np.array([
    [f_x, 0, cx],
    [0, f_y, cy],
    [0, 0, 1]
], dtype=np.float32)

# Coeficientes de distorsión
dist_coeffs = np.zeros((4, 1))



#PnP Solve
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs)

if success:
    print("Rotation:", rvec)
    print("Translation:", tvec)
    
    # Proyectar puntos 3D
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    print("Proyected points:", projected_points)
else:
    print("Cant solve PnP")
    