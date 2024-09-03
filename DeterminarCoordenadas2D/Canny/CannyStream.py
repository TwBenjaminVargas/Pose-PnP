#Codigo que tiene como finalidad la implementacion de canny para la deteccion de las coordenadas de los puntos utilizando
#primero una deteccion de contornos con Canny para mayor presicion pero mayor coste computacional
#Luego se planificaba realizar un calculo del centroide de dichos perimetros
#CODIGO INCOMPLETO - DESCONTINUADO DESDE 03/09/24
#Ultima version -> detectar bordes canny y determinar figura en cuestion
#Se deja en repositorio ya que con simples modificaciones se logra la funcionalidad deseada
import cv2 as cv
import cv2
import numpy as np

#iniciamos la camara
camera = cv.VideoCapture(0)
if not camera.isOpened():
    print("No se pudo abrir la camara")
    exit()

#iniciamos stream
while True:
    ret, frame = camera.read()
    if not ret:
        print("No se reciben frames")
        break
    
    
    #procesado de imagen
    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #filtro gaussiano para eliminacion de ruido (matriz impar para funcionamiento optimo) 0 para 
    #determinar automaticamente valor de campana de gauss
    gaussiana = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    #aplicamos deteccion de bordes con canny (ajustar umbral maximo)
    canny = cv2.Canny(gaussiana, 50, 350)
    
    #ALTERNATIVA A CANNY mas rapida en procesamiento y directa
    # Aplicar umbral (Thresholding)
    _, binary_image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    
    #buscamos contornos unicamente, copia de canny ya que modifica parametros, contornos externos del objeto unicamente cv2.RETR_EXTERNAL
    #metodo de aproximacion CHAIN_APPROX_SIMPLE elimina todos los puntos redundantes.
    #(CHAIN_APPROX_NONE toma todos los puntos pero mayor costo computacional)
    (contornos, jerarquia) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #mostrar cantidad de objetos encontrados (contornos)
    print("He encontrado {} objetos".format(len(contornos)))
    #Dibujado de contornos en imagen
    #imagen a dibujar
    #lista de contornos
    #numero de contornos a dibujar (todos con -1)
    #color
    #grosor
    cv2.drawContours(frame,contornos,-1,(0,0,255), 2)
    
    
    
    #obtener coordenadas en imagen de los contornos
    # Iterar sobre cada contorno y sus puntos
   # for contour in contornos:
        #for point in contour:
           # x, y = point[0]
            #print(f"Coordenadas del punto: x={x}, y={y}")
            #i+=1
            
    ## Iterar sobre cada contorno
    for contour in contornos:
    # Calcular la aproximación del contorno
        epsilon = 0.02 * cv2.arcLength(contour, True) #margen de error epsilon es un valor en píxeles que representa la distancia máxima permitida entre el contorno original y su versión simplificada.
        approx = cv2.approxPolyDP(contour, epsilon, True)       
            
        #filtrado de patron elegido
        if len(approx) == 3:
            print("El contorno es un triángulo")
        elif len(approx) == 4:
            print("El contorno es un cuadrado o rectángulo")
        elif len(approx) > 10:
            print("El contorno es un círculo")
        
    

    #mostrar imagen
    cv.imshow('camera',frame)
    #mostrar imagen en escala de grises
    cv.imshow('escala de grises',grayscale)
    #mostrar imagen con filtro gaussiano
    cv.imshow('gaussiana',gaussiana)
    #mostrar bordes canny
    cv.imshow('canny',canny)
    
    #mostrar bordes ubral (Alternativa mas rapida)
    cv.imshow('ThresHolding',binary_image)
    if cv.waitKey(1) == ord('q'):
        break
    
#liberamos la camara
camera.release()
cv.destroyAllWindows()