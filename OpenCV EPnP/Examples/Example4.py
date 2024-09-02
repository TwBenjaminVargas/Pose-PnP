import cv2
import numpy as np

# Definir el tamaño del papel en cm y la escala para convertirlo a píxeles
paper_width_cm = 10
paper_height_cm = 10
scale = 10  # 1 cm = 10 píxeles en la imagen (ajustar según sea necesario)

# Definir los puntos 3D en el sistema de coordenadas del objeto (esquinas del papel)
object_points = np.array([
    [0, 0, 0],                             # Esquina inferior izquierda
    [paper_width_cm * scale, 0, 0],        # Esquina inferior derecha
    [paper_width_cm * scale, paper_height_cm * scale, 0],  # Esquina superior derecha
    [0, paper_height_cm * scale, 0]        # Esquina superior izquierda
], dtype=np.float32)

# Configuración de la cámara
camera_matrix = np.array([[1000, 0, 320], 
                          [0, 1000, 240], 
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Coeficientes de distorsión

# Crear un objeto de captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar la imagen.")
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar el papel blanco usando un rango de colores en HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 200])  # Rango para blanco (ajustar si es necesario)
    upper_color = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    edges = cv2.Canny(mask, 50, 150)

    # Encontrar los contornos en la imagen de borde
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Asegurarse de encontrar al menos un contorno
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:  # Si se detectan 4 esquinas
            corners = np.array([point[0] for point in approx], dtype=np.float32)

            # Asegurarse de que los puntos están en el formato correcto
            image_points = np.array(corners, dtype=np.float32)

            # Resolver PnP usando EPnP
            success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

            if success:
                # Convertir el vector de rotación a una matriz de rotación
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Definir los ejes del sistema de coordenadas del objeto
                axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3) * scale

                # Proyectar los puntos 3D del eje en la imagen 2D
                imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                # Dibujar los ejes en la imagen
                corner = tuple(np.int0(image_points[0]))  # Esquina superior izquierda
                imgpts = np.int0(imgpts)
                frame = cv2.line(frame, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)  # Eje X (Rojo)
                frame = cv2.line(frame, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Eje Y (Verde)
                frame = cv2.line(frame, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)  # Eje Z (Azul)

    # Mostrar la imagen con los ejes proyectados
    cv2.imshow('Pose Estimation', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
