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
    
    # Iterar sobre cada contorno
    for contour in contornos:
    # Calcular la aproximación del contorno
        epsilon = 0.02 * cv2.arcLength(contour, True) # Margen de error epsilon
        approx = cv2.approxPolyDP(contour, epsilon, True)       
            
        # Filtrado de patron elegido
        if len(approx) == 3:
            print("El contorno es un triángulo")
        elif len(approx) == 4:
            print("El contorno es un cuadrado o rectángulo")
        elif len(approx) > 10:
            print("El contorno es un círculo")
        
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
    
    
    
