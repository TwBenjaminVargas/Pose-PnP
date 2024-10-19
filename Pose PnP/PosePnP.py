import cv2
import numpy as np

def cameraCalibration(fx = 1000,fy = 1000, x = 320, y = 240,d1 = 4,d2 = 1):
    # Matriz intrínseca
    K = np.array([
        [fx, 0, x],
        [0, fy, y],
        [0, 0, 1]
    ], dtype=np.float32)
    # Coeficientes de distorsión
    dist_coeffs = np.zeros((d1, d2))
    return K,dist_coeffs
    

    
# Implementa metodo PnP iterativo (por defecto)
def IterativePnP(object_points,image_points,K,dist_coeffs):
    
    try: 
        return cv2.solvePnP(object_points, image_points, K, dist_coeffs)
    except: #cantidad de puntos insuficientes        
        # Usar ExtrinsicGuess con estimacion
        rvec_initial = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec_initial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return cv2.solvePnP(object_points, image_points, K, dist_coeffs, rvec_initial, tvec_initial, True, cv2.SOLVEPNP_ITERATIVE)


# MAIN

# Iniciamos la camara
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("No se pudo abrir la camara")
    exit()

# CALIBRACION CAMARA
#Se encuentrar seteados los valores por defecto
K,dist_coeffs = cameraCalibration()

# Inicia stream de video
while True:
    ret, frame = camera.read()
    if not ret:
        print("No se reciben frames")
        break
    
# *****************************************************************************
#                               Filtrado
#******************************************************************************

    # Imagen en escala de grises
    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral (Thresholding)
    _, binary_image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    
    
# *****************************************************************************
#                           Localizacion de puntos
#******************************************************************************
    
    # Encontramos contornos usando findCountours
    (contornos, jerarquia) = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # lista para el registro de coordenadas
    coordenades = []
    
    #Dibujado de contornos en imagen
    cv2.drawContours(frame,contornos,-1,(0,0,255), 2)
    
    # Recorrer cada contorno y calcular su centroide
    i = 0
    for contorno in contornos:
        # Calcular los momentos del contorno
        M = cv2.moments(contorno)
    
        # Calcular las coordenadas del centroide usando los momentos
        # m10 suma ponderada con la intesidad del pixel (1 para los contornos), m00 area del contorno
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) # x
            cy = int(M["m01"] / M["m00"]) # y
        else:
            cx, cy = 0, 0
    
        # Dibujar el centroide en la imagen
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    
        # Marcar el centroide
        cv2.putText(frame, f"{i}: ({cx}, {cy})", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i+= 1
        #Almacenar en lista coordenadas 2D de la imagen
        coordenades.append((cx,cy))
        
    # Mostrar cantidad de objetos encontrados (contornos)
    print("He encontrado {} objetos".format(len(contornos)))
    
    
# *****************************************************************************
#                               Solve PnP
#******************************************************************************
    try:
        # 2D image coordenades (de abajo hacia arriba del patron)
        image_points = np.array([
        [coordenades[0][0], coordenades[0][1]],# P0
        [coordenades[1][0], coordenades[1][1]],# P1
        [coordenades[2][0], coordenades[2][1]] # P2
        ], dtype=np.float32)

        # 3D object coordenades
        object_points = np.array([
            [0, 0, 0],    # P0
            [0, 4, -2],    # P1
            [0, 12, 5],    # P2
        ], dtype=np.float32)

        success, rvec, tvec = IterativePnP(object_points,image_points,K,dist_coeffs)

        if success:
            print("Rotation:", rvec)
            print("Translation:", tvec)

            # Puntos 3D del objeto (aquí, los ejes de coordenadas)
            axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, 10]]).reshape(-1, 3)
            # Proyectar los puntos en el espacio 2D (Considerando su rotacion)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)

            # Convertir las coordenadas a enteros
            imgpts = np.int32(imgpts).reshape(-1, 2)
            image_points = np.int32(image_points).reshape(-1, 2)

            # Dibujar los ejes de coordenadas (Trazar linea entre los puntos)
            frame = cv2.line(frame, tuple(image_points[0].ravel()), tuple(imgpts[0].ravel()), (255, 0, 0), 7)  # Eje X en rojo
            frame = cv2.line(frame, tuple(image_points[0].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 7)  # Eje Y en verde
            frame = cv2.line(frame, tuple(image_points[0].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 7)  # Eje Z en azul
        else:
            print("Cant solve PnP")
    except:
        print("Pocos puntos detectados")

    # Mostrar imagen
    cv2.imshow('camera',frame)
    # Mostrar Threshold
    cv2.imshow('Threshold',binary_image)
    
    # Orden q para salir
    if cv2.waitKey(1) == ord('q'):
        break
    
# Liberamos la camara
camera.release()
cv2.destroyAllWindows()