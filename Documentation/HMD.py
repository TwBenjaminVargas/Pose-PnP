#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------
# importacion de los modulos
# ----------------------

import sys, getopt
import cv2
import numpy as np
import os
import threading
import pygame
import time
from MPU9250Driver import MPU
# ----------------------
print('Version de Python: ',(".".join(map(str, sys.version_info[:3]))))  
print('Version de OpenCV: ', cv2.__version__)
print('Version de PyGame: ', pygame.__version__)

#Valor donde se guarda la pocicion actual del IMU

bloqueo_IMU = threading.Lock()

#Variable global donde se va a guardar la pocicion de la cabeza en la terna roll pich y yaw

Situacion = np.array((0,0,0))
bloqueo_Situacion = threading.Lock()

DeribaGyro = [0,0,0]

# -------------------
# constantes
# ------------------

WIDTH = 800
HEIGHT = 600

OffSetX = 100
OffSetY = 0

#Configuro variable global para hacer de interface a frame

Frame = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
bloqueo_Frame = threading.Lock()

#Variable Global para hacer de interface para la telemetria del video que capturan las webcam a una terminal remota

Frame_str = []
bloqueo_Frame_str = threading.Lock()


'''Redefinicion de funciones matenaticas para usarlas en grados.
Es deceable poder implementar en futuro optimizaciones como una serie de lorenz para mejorar la velocidad.'''
def sin(grados):
    return np.sin(np.radians(grados))

def cos(grados):
    return np.cos(np.radians(grados))

def tan(grados):
    return np.tan(np.radians(grados))

def arctan2(x,y):
    return np.rad2deg(np.arctan2(x,y))

'''Lista de tuplas que contiene todos los objetivos a marcar.
   Estan expresado en gados. Roll, Pitch y Yaw repectivamete.'''   
Objetivos = [(0,0,0), (0,0,15), (0,0,-15), (0,15,0), (0,-15,0)]

'''Lista de objetivos dibujados solo para propocitos de debub'''
Lista_c = []

def display_init():
    '''Funcion para Configurar e iniciar la pantalla'''
    pygame.display.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF, vsync=1)
    #print(pygame.display.Info())
    return screen

retardo_update_display = float()  
def update_display():
    """Trasfomo Frame en un objeto de pygame y lo escribo en el FrameBuffer
    utilizar un semaforo para bloquear a Frame mientras se lee"""
    global retardo_update_display
    global bloqueo_Frame
    # greyscale Palette
    grey_palette = [(0, 0, 0)]
    for i in range(1, 256):
        grey_palette.append( (i, i, i) )
    tiempo_0 = float()
    screen = display_init()
    while True:
        '''En teoria segun la libreria pygame.display.flip() debiera esperar al VSync por estar accediendo al FB
        https://www.pygame.org/docs/ref/display.html#pygame.display.flip'''
        tiempo_0 = time.perf_counter()
        # Limpio el display
        screen.fill((0,0,0))
        # Copio la variable Frame a una imagen de pygame
        with bloqueo_Frame:
            pgFrame = pygame.image.frombuffer(Frame, (WIDTH,HEIGHT), 'P')
        pgFrame.set_palette(grey_palette)
        #Escribo la pantalla
        screen.blit(pgFrame, (0,0))
        #Cambio el cuadro
        pygame.display.flip()
        retardo_update_display =  time.perf_counter() - tiempo_0

def dibujarHorizonte(grados, horizonte_img):
    global Frame
    (h, w) = horizonte_img.shape[:2]
    
    center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, grados, 1)
    rotated90 = cv2.warpAffine(horizonte_img, M, (h, w))
    
    porcion_fondo = Frame[int(WIDTH/2):int((HEIGHT-rotated90.shape[0])),int((WIDTH/2)-(rotated90.shape[0]/2)):int((WIDTH/2)+(rotated90.shape[0]/2))]
    
    dst = cv2.addWeighted(rotated90,1,porcion_fondo,1,0)
    
    Frame[int(WIDTH/2):int((HEIGHT-rotated90.shape[0])),int((WIDTH/2)-(rotated90.shape[0]/2)):int((WIDTH/2)+(rotated90.shape[0]/2))] = dst
    
    return True

def dibujarObjetivos(Situacion_local):
    '''no lo tengo claro aun'''
    global Frame
    global Lista_c
    Lista_c.clear()
    
    k = 300 / tan(10) #Pixeles dividido tangente del FOV
    for objetivo in Objetivos:
        objetivo_Proyectado = np.array(objetivo) - Situacion_local * [1,1,-1] # La matris invierte un eje
        if np.all(np.abs(objetivo_Proyectado)[1:] < [10,10]):
            '''Si el objetivo esta en el campo de vision'''
            c = tuple(np.round(tan(objetivo_Proyectado[1:]) * k + np.array(Frame.shape)/2)[::-1].astype(int))
            #[1:] me quedo con los elemntos apartir de 1, [::-1] invierto el orden de los elemntos
            cv2.drawMarker(Frame, c, 255, cv2.MARKER_TRIANGLE_UP, 25, 3)
            Lista_c.append(c)
            
retardo_update_frame = float()  
def update_frame():
    """Construye un nuevo Frame.
    Se debe ejecura ante los siguientes eventos:
                                Existe una nueva actitud.
                                Se cambio o agrego un objetivo.
                                Se cambio o agrego informacion de pantalla.
    Se debe bloquear Frame mientras se los escribe.
    horizonte: grados de la cabeza respecto al horizonte.
    objetivos: Lista de tuplas en theta, phi de cada uno de los objetivos respecto de la cabeza."""
    global Frame
    global bloqueo_Frame
    global retardo_update_frame
    tiempo_0 = float()
    #Cargo imágenes
    fondo_img = cv2.imread(os.path.join('/home/pi/fondo2.png'),cv2.IMREAD_GRAYSCALE)
    horizonte_img = cv2.imread(os.path.join('/home/pi/horizonte2.png'),cv2.IMREAD_GRAYSCALE)
    #objetivo_img = cv2.imread(os.path.join('/home/pi/Objetivo.png'),cv2.IMREAD_GRAYSCALE)
    font = cv2.FONT_ITALIC
    while True:
        '''con los datos de Situacion, objetivos y demas vedura costruyo un nuevo frame.'''
        tiempo_0 = time.perf_counter()
        with bloqueo_Frame:
            Frame[0:fondo_img.shape[0], 100:100+fondo_img.shape[1]] = fondo_img
            with bloqueo_Situacion:
                Situacion_local = Situacion
            cv2.putText(Frame, '%4.1f' % Situacion_local[1], (360,90), font, 1, 255, 2, cv2.LINE_AA)
            cv2.putText(Frame, '%4.1f' % Situacion_local[2], (360,520), font, 1, 255, 2, cv2.LINE_AA)
            #cv2.putText(Frame, 'Situacion del casco: roll:%4.1f pich:%4.1f yaw:%4.1f' % tuple(Situacion_local), 
            #            (110,400), font, 0.5, 255, 1, cv2.LINE_AA)
            dibujarHorizonte(Situacion_local[0], horizonte_img)
            dibujarObjetivos(Situacion_local)
            Frame = cv2.flip(Frame, 1) #invierto la imagen por la optica.

        retardo_update_frame =  time.perf_counter() - tiempo_0
        '''Si los Frame salen cada 15 ms o 60 hz y tarde menos que eso puedo esperar para construir el proximo. 
        no es un metodo muy eficiente ni preciso ya que no mantengo ningun tipo de sincronimo con la pantalla.
        '''
        if retardo_update_frame < 15/1000:
            time.sleep(15/1000-retardo_update_frame)

retardo_update_Situacion = float()        
evento_IMU = threading.Event()
def read_IMU():
    """Lee de forma continua el IMU.
    Cuando existe un nuevo dato debe exitar el evento evento_IMU
    Actualiza IMU con un np.array en grados de (roll, pich, yaw)
    Se debe bloquear IMU en la escritura.
    """
    global Situacion
    global hilo_update_refencia_optica
    global retardo_update_Situacion
    tiempo_0 = float()
    gyro = 2000      # 250, 500, 1000, 2000 [deg/s]
    acc = 4         # 2, 4, 7, 16 [g]
    mag = 16        # 14, 16 [bit]
    tau = 0.98
    global mpu 
    mpu = MPU(gyro, acc, mag, tau)

    # Set up the IMU and mag sensors
    mpu.setUpIMU()

    mpu.setUpMAG()

    mpu.setUpResetIntegrator()
    # Calibrate the mag or provide values that have been verified with the visualizer
    #mpu.calibrateMagGuide()
    bias = [213.847, 366.788, -253.124]
    scale = [1.011, 0.978, 1.011]
    mpu.setMagCalibration(bias, scale)
    # Calibrate the gyro with N points
    mpu.calibrateGyro(1000)
    mpu.calibrateMag(100)
    while True:
        tiempo_0 = time.perf_counter()
        mpu.compFilter()
        with bloqueo_Situacion:
            Situacion = np.array((mpu.roll, mpu.pitch, mpu.yaw))
        mpu.readRawMag()
        retardo_update_Situacion =  time.perf_counter() - tiempo_0

def colineal_equidistante(x_values, y_values, margen, tamaño):
    """
    Metodo de deteccion por area de un triangulo muy poco eficiente
    https://davidgarciafer.github.io/Colinear-Points/***
    """
    size = len(x_values)
    arreglo_x = []
    arreglo_y = []
    for i in range(size):
        x1, y1 = x_values[i], y_values[i]
        for j in range(i, size): 
            if i == j:
                continue
            x2, y2 = x_values[j], y_values[j]
            for k in range(j, size):
                if i == k or j == k:
                    continue
                x3, y3 = x_values[k], y_values[k]
                # Comprobamos si el area es nula
                area = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))/2
                distancias = []
                distancias.append(np.sqrt((x1-x2)**2 + (y1-y2)**2))
                distancias.append(np.sqrt((x2-x3)**2 + (y2-y3)**2))
                distancias.append(np.sqrt((x1-x3)**2 + (y1-y3)**2))
                distancias.sort()
                '''Diferencia de distancia entres los dos segmentos mas cortos'''
                diferencia1 = np.abs(distancias[0] - distancias[1])
                '''Diferencia de distancia entre el segmento mas largo y lo que se espera que tenga el patron'''
                diferencia2 = np.abs(distancias[2] - tamaño)
                if area < margen**2 and diferencia1 < margen and diferencia2 < margen*2:
                    arreglo_x += x1, x2, x3
                    arreglo_y += y1, y2, y3
                    
                    #El centro se toma como el promedio de los tres centros
                    arreglo_x.append(int(np.mean([x1,x2,x3])))
                    arreglo_y.append(int(np.mean([y1,y2,y3])))
                    
                    #Para calcular el angulo verifico el orden en que aparecen los punto no es una forma muy elegante :( 
                    if x1 < x2 < x3:
                        arreglo_x.append(np.mean((
                            np.rad2deg(np.arctan2(x2-x1,y2-y1)), 
                            np.rad2deg(np.arctan2(x3-x1,y3-y1)), 
                            np.rad2deg(np.arctan2(x3-x2,y3-y2))
                            )))
                        #print('caso 1')
                    elif x1 < x3 < x2:
                        arreglo_x.append(np.mean((
                            np.rad2deg(np.arctan2(x2-x1,y2-y1)),
                            np.rad2deg(np.arctan2(x3-x1,y3-y1)),
                            np.rad2deg(np.arctan2(x2-x3,y2-y3))
                            )))
                        #print('caso 2')
                    elif x2 < x1 < x3:
                        arreglo_x.append(np.mean((
                            np.rad2deg(np.arctan2(x1-x2,y1-y2)),
                            np.rad2deg(np.arctan2(x3-x1,y3-y1)),
                            np.rad2deg(np.arctan2(x3-x2,y3-y2))
                            )))
                        #print('caso 3')
                    elif x2 < x3 < x1:
                        arreglo_x.append(np.mean((
                            np.rad2deg(np.arctan2(x1-x2,y1-y2)),
                            np.rad2deg(np.arctan2(x1-x3,y1-y3)),
                            np.rad2deg(np.arctan2(x3-x2,y3-y2))
                            )))
                        #print('caso 4')
                    elif x3 < x1 < x2:
                        arreglo_x.append(np.mean((
                            np.rad2deg(np.arctan2(x2-x1,y2-y1)),
                            np.rad2deg(np.arctan2(x1-x3,y1-y3)),
                            np.rad2deg(np.arctan2(x2-x3,y2-y3))
                            )))
                        #print('caso 5')
                    elif x3 < x2 < x1:
                        arreglo_x.append(np.mean((
                            np.rad2deg(np.arctan2(x1-x2,y1-y2)),
                            np.rad2deg(np.arctan2(x1-x3,y1-y3)),
                            np.rad2deg(np.arctan2(x2-x3,y2-y3))
                            )))
                        #print('caso 6')           
                    else:
                        arreglo_x.append(0)
                        #print('caso no contemplado')
                    arreglo_y.append(distancias[2])
                    #print('Se alinearon los astros.')
                    #print('Area del triangulo: ' + str(area))
                    #print('Diferencia de distancia entre Segmentos mas cortos: ' + str(int(diferencia1)))
                    #print('Diferencia de distancia entre Segmento mas largo y el tamaño esperado: ' + str(int(diferencia2)))                    
    #print(centros)
    #print(angulos)
    return arreglo_x, arreglo_y

retardo_update_refencia_optica = float()
def update_refencia_optica():
    tiempo_0 = float()
    global retardo_update_refencia_optica
    global Frame_str
    global DeribaGyro
    '''Contro de WEBCam'''
    cap = cv2.VideoCapture(0,cv2.CAP_V4L2)#0 corresponde a /dev/video0, backend V4L2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cap.set(cv2.CAP_PROP_FPS, 5)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    Frame_str = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3),dtype=np.uint8)

    ####VALORES QUE SE CREE VA A TENER EL LED AL ARRANCAR EL PROGRAMA####
    
    #areas en pixeles
    area_frame = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    areaminima= 40
    areamaxima= 270
    
    #margen de linealidad
    margen = 20
    tamaño = 400
    #########################################################
    
    if (cap.isOpened()): #devuelve verdadero si la captura del video ya se ha inicializado
        pass#print('Capturando video para referencia optica de: camara por defecto')
    else:
        #print("Error abriendo la captura de video")
        exit()
  
    cap.set(cv2.CAP_PROP_CONTRAST, 128)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_SATURATION, 128)
    cap.set(cv2.CAP_PROP_ZOOM, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 30) # 0 cerca 254 lejos
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 desabilitado 3 habilitado
    cap.set(cv2.CAP_PROP_EXPOSURE, 60) # 0 corto tiempo de expocicion 2048 largo tiempo de expocicion
    cap.set(cv2.CAP_PROP_AUTO_WB, 1) # 0 Deshabilita el balance de blancos
    #cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
    cap.set(cv2.CAP_PROP_GAIN,100)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    
    #Reinicio el temporisador para calibrar los giroscopos
    mpu.dtTimer2 = time.perf_counter()
    while(True):
        '''PROP_CAP = cv2.CAP_PROP_BRIGHTNESS
        if cap.get(PROP_CAP) < 254:
            cap.set(PROP_CAP, cap.get(PROP_CAP) + 1)
        else:
            cap.set(PROP_CAP, 1)
        print(cap.get(PROP_CAP))
        Ver el link 
        https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
        '''
        umbral_bajo = 45
        umbral_alto = 170
        '''
        if V < 190:
            V += 1
        else:
            V = 10
        
        umbral_bajo = int(V)
        umbral_alto = int(V) +2
        print(umbral_bajo)
        '''
       
        tiempo_0 = time.perf_counter()
        '''grabo el valor de los giroscopos para calcular la deriba al momento de capturar la imagen.
         sino estaria intruciendo un error por el tiempo que tardo en procesar la imagen.'''
        Giroscopos_tf = (mpu.gyroRoll,mpu.gyroPitch,mpu.gyroYaw)
        cap.grab() #retorna un bool se puedu usar para controlar el while
        '''aqui le ordena a la camara capturar la imagen,
        se va a demorar hasta que este capturada  si la camara ya la capturo lo devuelve de imediato , 
        lo importante es que la marca de tiempo al salir del cuadro es precisa
        https://medium.com/@gastonace1/detecci%C3%B3n-de-objetos-por-colores-en-im%C3%A1genes-con-python-y-opencv-c8d9b6768ff
        
        '''
        
        # Captura marco por marco 
        ret, frame = cap.retrieve()
        '''Aqui recibe la imagen desde la camara, es un cuadro que ya se capturo previamente y quedo en el buffer'''
                    
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gauss = cv2.GaussianBlur(frame_gray,(3,3),0)
        frame_mask = cv2.inRange(frame_gauss, umbral_bajo, umbral_alto)
        try:
            Frame_str = frame#cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2RGB)
        except:
            pass
        #Calculo los contronos
        contornos, jerarquia = cv2.findContours(frame_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(Frame_str, contornos, -1, (0,255,0), 1)
        
        referencias_LEDx = []
        referencias_LEDy = []

        for contorno in contornos:#recorro todos los contornos encontrados
            area=cv2.contourArea(contorno)#calculo el area del controno i
            (x, y), radio = cv2.minEnclosingCircle (contorno)#encierro en el minimo circulo posible el contorno
            centro = (int (x), int (y))
            radio = int (radio)
            cv2.putText(Frame_str, str(area), tuple(sum(x) for x in zip(centro, (-20,60))), cv2.FONT_ITALIC,1, (0,0,255), 1, cv2.LINE_AA)
            if(areaminima <area< areamaxima):#filtro por area
                cv2.drawMarker(Frame_str, centro, 255, cv2.MARKER_DIAMOND, 50, 2)
                referencias_LEDx.append(int(x))
                referencias_LEDy.append(int(y))
        # arreglo_x : x1,x2,x3,cx,a,...
        # arreglo_y : y1,y2,y3,cy,tamaño,...

        arreglo_x, arreglo_y = colineal_equidistante(referencias_LEDx, referencias_LEDy, margen, tamaño)
        
        for i in range(0,len(arreglo_x),5):
            cv2.line(Frame_str,(arreglo_x[i],arreglo_y[i]),(arreglo_x[i+1],arreglo_y[i+1]),(255,255,0),2,cv2.LINE_AA)
            cv2.line(Frame_str,(arreglo_x[i],arreglo_y[i]),(arreglo_x[i+2],arreglo_y[i+2]),(255,255,0),2,cv2.LINE_AA)
            cv2.line(Frame_str,(arreglo_x[i+1],arreglo_y[i+1]),(arreglo_x[i+2],arreglo_y[i+2]),(255,255,0),2,cv2.LINE_AA)
            cv2.drawMarker(Frame_str, (arreglo_x[i+3], arreglo_y[i+3]), (128,128,128), cv2.MARKER_STAR, 30, 2)
            cv2.putText(Frame_str, str(arreglo_x[i+4]), (arreglo_x[i+3]-30,arreglo_y[i+3]-30), cv2.FONT_ITALIC,1, (255,255,0), 1, cv2.LINE_AA)
        '''Bucle de control realimentado.
         Se tendria que controlar:
                                margen
                                tamaño
                                area minima y maxima 
                                anda maaaalll
                                
        '''
        if len(arreglo_x) < 5 and margen < 40:
            margen += 1
        else:
            margen -= 1
        #print(margen)        
        if len(arreglo_x) == 5:
            #Se cumplieron todas las condiciones para un solo partron
            tamaño = arreglo_y[4]
            #realizo el calculo de la reriba acumulada
            DeribaGyro = Giroscopos_tf - np.array((0,0,0))
            mpu.updateCalibrateGyro(DeribaGyro)
            print(DeribaGyro)
        retardo_update_refencia_optica =  time.perf_counter() - tiempo_0

def telemetria_consola():
    while True:
        os.system('clear')
        #print('Situacion del casco: roll:\t%4.1f pich:\t%4.1f yaw:\t%4.1f' % tuple(Situacion)) 
        #print('IMU del casco: roll:\t%4.1f pich:\t%4.1f yaw:\t%4.1f' % (mpu.gyroRoll,mpu.gyroPitch,mpu.gyroYaw)) 
        print('Para actualizar Frame se demora %3.0f ms' % (retardo_update_frame*1000))
        print('Se acutulariza Display cada %3.0f ms' % (retardo_update_display*1000))
        print('Se acutulariza situacion cada %1.2f ms' % (retardo_update_Situacion*1000))
        #print('Se capturan imagenes para referencia optica cada %3.0f ms' % (retardo_update_refencia_optica*1000))
        #print('Coordenadas de Objetivos dibujados',Lista_c)
        #print('Magnetometro: x:%4i y:%4i z:%4i' % (mpu.mx, mpu.my, mpu.mz))

        time.sleep(1)

def telemetria_video():
    '''
    mjpeg codec 
    Tranmisor:
    sudo gst-launch-1.0 v4l2src -v  ! video/x-raw,width=1920,height=1080,format=BGR ! videoconvert ! video/x-raw,format=YUY2 ! jpegenc idct-method=ifast ! queue ! rtpjpegpay pt=127 ! udpsink host=192.168.16.15 port=5000
    
    Reseptor :
    gst-launch-1.0 -v udpsrc port=5000 ! "application/x-rtp,media=(string)video,encoding-name=(string)JPEG" ! rtpjpegdepay ! jpegdec ! xvimagesink sync=false
    '''
    gst_str_rtp = 'appsrc ! videoconvert ! video/x-raw,format=YUY2 ! queue ! jpegenc ! jpegparse ! rtpjpegpay ! udpsink host=192.168.16.109 port=5000'
    fps = 10 #VideoWriter matiene la tasa 
    out = cv2.VideoWriter(gst_str_rtp, cv2.CAP_GSTREAMER, 0, fps, (Frame_str.shape[1], Frame_str.shape[0]), True)
    while True:
        out.write(Frame_str)
 
'''Creo los hilos'''
hilo_read_IMU = threading.Thread(name='read_IMU', target=read_IMU)
hilo_telemetria_consola = threading.Thread(name='telemetria', target=telemetria_consola)
hilo_telemetria_video = threading.Thread(name='telemetria_video', target=telemetria_video)
hilo_update_frame = threading.Thread(name='update_frame', target=update_frame)
hilo_update_display = threading.Thread(name='update_display', target=update_display)
hilo_update_refencia_optica = threading.Thread(name='update_refencia_optica', target=update_refencia_optica)

hilo_telemetria_consola.setDaemon(True)
hilo_telemetria_video.setDaemon(True)
hilo_read_IMU.setDaemon(True)
hilo_update_display.setDaemon(True)
hilo_update_frame.setDaemon(True)
hilo_update_refencia_optica.setDaemon(True)

'''Inicio de algo ;-('''
hilo_read_IMU.start()
time.sleep(1)
#hilo_update_refencia_optica.start()

hilo_update_frame.start()
hilo_update_display.start()

time.sleep(5)
hilo_telemetria_consola.start()
#hilo_telemetria_video.start()
while True:
    time.sleep(1)
