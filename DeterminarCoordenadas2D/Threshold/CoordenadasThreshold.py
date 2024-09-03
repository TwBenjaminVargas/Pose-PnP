#Utilizamos la alternativa a Canny para la deteccion de bordes, con el obbjetivo de una mayor
#velocidad de ejecucion y costo computacional, ya que correra en un dispositivo embebido.
#Threshold es directo y simple, se basa en un umbral para la determinacion de bordes
#decidiendo entre color blanco y negro de los pixeles segun corresponda.
#Se lo considera optimo dado que trabajamos con luces infrarrojas que proveen de alta intensidad a los pixeles.
#Se evita el uso de filtros para hacer el proceso aun mas optimo

import cv2 as cv
import cv2
import numpy as np

# Iniciamos la camara
camera = cv.VideoCapture(0)
if not camera.isOpened():
    print("No se pudo abrir la camara")
    exit()
    
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

    
        
    # Mostrar imagen
    cv.imshow('camera',frame)
    # Mostrar Threshold
    cv.imshow('Threshold',binary_image)
    
    # Orden q para salir
    if cv.waitKey(1) == ord('q'):
        break
    
# Liberamos la camara
camera.release()
cv.destroyAllWindows()
    
    
    
