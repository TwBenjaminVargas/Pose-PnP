# EN DESARROLLO
# Determinacion de la pose utilizando "Vincha con luces infrarojas" adicionalmente y para reduccion de complicaciones con 
# filtros, se cuenta con filtro fisico de infrarojo
# Se utilizara threshold y un proceso de calculo de centroide para determinar de manera dinamica las coordenadas
# 2D (de la imagen) de interes para la ejecucion del metodos PnP default de la libreria OpenCV

import cv2
import numpy as np

# Iniciamos la camara
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("No se pudo abrir la camara")
    exit()

# Calibracion de la camara

# Parámetros intrínsecos de la cámara (valores aproximados)
f_x = 1000
f_y = 1000
x = 320
y = 240

# Matriz intrínseca
K = np.array([
    [f_x, 0, x],
    [0, f_y, y],
    [0, 0, 1]
], dtype=np.float32)

# Coeficientes de distorsión
dist_coeffs = np.zeros((4, 1))
    
# Iniciamos stream
while True:
    ret, frame = camera.read()
    if not ret:
        print("No se reciben frames")
        break
    
    # Imagen en escala de grises
    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral (Thresholding)
    _, binary_image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    
    # Encontramos contornos usando findCountours
    (contornos, jerarquia) = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Mostrar cantidad de objetos encontrados (contornos)
    print("He encontrado {} objetos".format(len(contornos)))
    
    #Dibujado de contornos en imagen
    cv2.drawContours(frame,contornos,-1,(0,0,255), 2)
    
    # Recorrer cada contorno y calcular su centroide
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
        cv2.putText(frame, f": ({cx}, {cy})", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        

    # Pose PnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs)
    
    
        
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
    
    
    
