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
    """
    Calcula la pose de un objeto utilizando el método iterativo de Perspective-n-Point (PnP).

    Parámetros:
    - object_points: Coordenadas 3D de los puntos del objeto en el espacio real. Es una matriz Nx3 (float).
    - image_points: Coordenadas 2D de los puntos correspondientes en la imagen. Es una matriz Nx2 (float).
    - K: Matriz de calibración de la cámara (matriz intrínseca) 3x3.
    - dist_coeffs: Coeficientes de distorsión de la lente de la cámara.

    Retorna:
    - rvec: Vector de rotación que describe la orientación del objeto en el espacio 3D.
    - tvec: Vector de traslación que describe la posición del objeto en el espacio 3D.

    Funcionalidad:
    - Si se proporciona una cantidad suficiente de puntos (al menos 4), se utiliza el método por defecto 
      `cv2.SOLVEPNP_ITERATIVE` para resolver el problema de PnP.
    - En caso de que la cantidad de puntos sea insuficiente para realizar la estimación, se intenta utilizar 
      el método `cv2.SOLVEPNP_ITERATIVE` con una suposición inicial (`ExtrinsicGuess`). En este caso, se usa 
      una estimación inicial para `rvec` (vector de rotación) y `tvec` (vector de traslación), que son inicializados 
      como `[0.0, 0.0, 0.0]` y `[0.0, 0.0, 1.0]` respectivamente.
      
    Excepciones:
    - En caso de error debido a una cantidad insuficiente de puntos, se proporciona una estimación inicial para 
      continuar el cálculo.

    """
    try: 
        return cv2.solvePnP(object_points, image_points, K, dist_coeffs)
    except Exception as e: #cantidad de puntos insuficientes        }
        print(e)
        # Usar ExtrinsicGuess con estimacion
        rvec_initial = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec_initial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return cv2.solvePnP(object_points, image_points, K, dist_coeffs, rvec_initial, tvec_initial, True, cv2.SOLVEPNP_ITERATIVE)

# Implementa Threshold para localizar los puntos
def applyThreshold(grayscale, umbral = 127, color = 255, operationtype = cv2.THRESH_BINARY):
    """
    Aplica un umbral (threshold) a una imagen en escala de grises para generar una imagen binaria o realizar 
    otras operaciones de segmentación.

    Parámetros:
    - grayscale: Imagen de entrada en escala de grises (matriz de intensidades).
    - umbral (int, opcional): Valor de umbral. Los píxeles con valores mayores o iguales a este umbral serán 
      considerados para la operación de umbral. Valor por defecto: 127.
    - color (int, opcional): Valor máximo que se asignará a los píxeles que cumplan con la condición de umbral.
      Por defecto es 255, lo que equivale al color blanco en una imagen binaria.
    - operationtype (int, opcional): Tipo de operación de umbral a aplicar. Por defecto es cv2.THRESH_BINARY.

    Tipos de operación (operationtype):
    - cv2.THRESH_BINARY: Los píxeles por debajo del umbral se convierten en 0 (negro), los que están por 
      encima o igual al umbral se convierten en color (valor máximo, por defecto 255, blanco).
    - cv2.THRESH_BINARY_INV: Inverso de THRESH_BINARY. Los píxeles por debajo del umbral se convierten en color 
      (blanco), y los por encima o igual al umbral se convierten en 0 (negro).
    - cv2.THRESH_TRUNC: Los píxeles por encima del umbral se truncarán al valor del umbral, los píxeles por 
      debajo del umbral no se modifican.
    - cv2.THRESH_TOZERO: Los píxeles por debajo del umbral se convierten en 0, los píxeles por encima o igual 
      al umbral no se modifican.
    - cv2.THRESH_TOZERO_INV: Inverso de THRESH_TOZERO. Los píxeles por encima o igual al umbral se convierten 
      en 0, los píxeles por debajo del umbral no se modifican.

    Retorna:
    - Una tupla (ret, imagen_binarizada), donde 'ret' es el valor del umbral usado y 'imagen_binarizada' es la 
      imagen resultante después de aplicar el umbral.
    """
    return cv2.threshold(grayscale, umbral, color, operationtype)

# Implemetar Canny 
def applyCanny(grayscale,inf = 50,sup = 350):
    """
    Aplica el detector de bordes Canny sobre una imagen en escala de grises.

    Parámetros:
    - grayscale: Imagen en escala de grises (numpy array) a la cual se le aplicará el detector de bordes Canny.
    - inf (int, opcional): Umbral inferior para la detección de bordes. Los píxeles con un gradiente
      por debajo de este valor no serán considerados como bordes. Valor por defecto: 50.
    - sup (int, opcional): Umbral superior para la detección de bordes. Los píxeles con un gradiente
      mayor que este valor serán considerados como bordes. Valor por defecto: 350.

    Retorna:
    - edges: Imagen binaria resultante (numpy array) donde los píxeles con valor 255 representan los bordes 
      detectados y los píxeles con valor 0 representan áreas sin bordes.

    Descripción:
    - El detector de bordes Canny es un algoritmo basado en la detección de gradientes en la imagen.
      Detecta bordes siguiendo estos pasos:
      1. **Gradiente de la imagen**: Calcula los cambios en la intensidad de los píxeles (gradiente).
      2. **Umbralización**: Los píxeles cuyo gradiente es mayor que el umbral superior se consideran 
         bordes fuertes y se marcan inmediatamente. Los píxeles cuyo gradiente está entre los dos umbrales
         se consideran bordes débiles, pero solo se marcan como bordes si están conectados a un borde fuerte.
    - Los parámetros `inf` y `sup` determinan cuán estricta es la detección de bordes. Umbrales más bajos 
      detectan más bordes, incluidos algunos ruidosos; umbrales más altos detectan menos bordes.

    Ejemplo de uso:
    ```python
    edges = applyCanny(grayscale_image, 100, 200)
    cv2.imshow('Bordes detectados', edges)
    ```

    Esta función es útil para la detección de contornos, segmentación de objetos y análisis de imágenes donde los bordes juegan un papel importante.
    """
    return cv2.Canny(grayscale, inf, sup)
# Obtener centroide de contornos y dibujarlo
def getCentroid(contornos, frame):
    """
    Calcula el centroide de cada contorno proporcionado y lo dibuja en la imagen.

    Parámetros:
    - contornos: Lista de contornos obtenidos previamente usando funciones como cv2.findContours(). 
      Cada contorno es un array de coordenadas que representan el contorno de un objeto en la imagen.
    - frame: Imagen en la que se dibujarán los centroides y las coordenadas correspondientes.

    Proceso:
    - Para cada contorno en la lista de contornos:
        1. Se calculan los momentos del contorno utilizando la función cv2.moments().
        2. Se calculan las coordenadas del centroide usando las siguientes fórmulas:
            - `cx = M["m10"] / M["m00"]`: Coordenada x del centroide.
            - `cy = M["m01"] / M["m00"]`: Coordenada y del centroide.
            Nota: Si el área del contorno (`m00`) es cero, se evita la división por cero y el centroide se establece en (0, 0).
        3. Se dibuja un círculo rojo en la posición del centroide utilizando cv2.circle().
        4. Se muestra el índice del contorno y las coordenadas del centroide usando cv2.putText().

    Retorna:
    - coordenades: Lista de tuplas con las coordenadas (cx, cy) de los centroides de cada contorno.

    Ejemplo de uso:
    - Esta función es útil para analizar los objetos detectados en una imagen, marcando visualmente 
      el centro de cada objeto detectado (usando contornos) y mostrando las coordenadas de los centros.
    """
     # lista para el registro de coordenadas
    coordenades = [] 
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
        cv2.putText(frame, f"{i}: ({cx}, {cy} A:{cv2.contourArea(contorno)})", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i+= 1
        #Almacenar en lista coordenadas 2D de la imagen
        coordenades.append((cx,cy))
    return coordenades

# Proyecta eje 3D en imagen
def proyectAxis(frame,rvec,tvec,K,dist_coeffs,image_points):
    """
    Proyecta los ejes de coordenadas 3D sobre la imagen 2D y dibuja las líneas de los ejes X, Y y Z.

    Parámetros:
    - frame: Imagen en la que se dibujarán los ejes de coordenadas.
    - rvec: Vector de rotación que describe la orientación del objeto en el espacio 3D.
    - tvec: Vector de traslación que describe la posición del objeto en el espacio 3D.
    - K: Matriz intrínseca de la cámara (parámetros de calibración).
    - dist_coeffs: Coeficientes de distorsión de la lente de la cámara.

    Proceso:
    1. Se definen tres puntos en 3D que representan los ejes de coordenadas:
       - (10, 0, 0): Punto en el eje X.
       - (0, 10, 0): Punto en el eje Y.
       - (0, 0, 10): Punto en el eje Z.
    2. Se proyectan estos puntos 3D a coordenadas 2D en la imagen usando la función cv2.projectPoints().
    3. Se trazan líneas en la imagen representando los ejes de coordenadas:
       - Eje X en rojo.
       - Eje Y en verde.
       - Eje Z en azul.

    Retorna:
    - frame: Imagen con los ejes X, Y, Z dibujados en sus posiciones proyectadas.

    Función útil:
    - Esta función es común en aplicaciones de visión por computadora, como realidad aumentada, donde es necesario visualizar la orientación y posición de un objeto en el espacio 3D con respecto a la cámara.

    Ejemplo:
    - La función proyecta los ejes de coordenadas sobre un objeto detectado y muestra cómo están orientados en la imagen.

    """
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
    
    # FILTRADO DE IMAGEN
    
    # Imagen en escala de grises
    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #filtro gaussiano para eliminacion de ruido (matriz impar para funcionamiento optimo) 0 para 
    #determinar automaticamente valor de campana de gauss
    gaussiana = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    
    # Aplicar umbral (Thresholding)
    #_, binary_image = applyThreshold(gaussiana)
    # Aplicar canny 
    binary_image = applyCanny(gaussiana)
    
    # LOCALIZACION DE PUNTOS
    
    # Encontramos contornos usando findCountours
    #(contornos, jerarquia) = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #(contornos, jerarquia) = cv2.findContours(binary_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    (contornos, jerarquia) = cv2.findContours(binary_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Definir el área máxima permitida para los contornos
    area_maxima = 120# Ajusta este valor según sea necesario
    area_minima = 0
    # Filtrar los contornos por área
    contornos_filtrados = [contorno for contorno in contornos if area_minima < cv2.contourArea(contorno) < area_maxima]
    
    #Dibujado de contornos en imagen
    cv2.drawContours(frame,contornos_filtrados,-1,(0,0,255), 2)
    
    # lista para el registro de coordenadas
    coordenades = getCentroid(contornos_filtrados,frame)
    print(coordenades) 
    # Mostrar cantidad de objetos encontrados (contornos)
    print("He encontrado {} objetos".format(len(contornos_filtrados)))
    
    # SOLVE PNP
    
    try:
        # Asegúrate de que la variable coordenades tenga al menos 3 elementos
        if len(coordenades) < 3:
                raise ValueError("No hay suficientes coordenadas 2D en 'coordenades'. Se requieren al menos 3 puntos.")
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
            [0, 12, 5],    # P24
        ], dtype=np.float32)

        success, rvec, tvec = IterativePnP(object_points,image_points,K,dist_coeffs)

        if success:
            print("Rotation:", rvec)
            print("Translation:", tvec)
            proyectAxis(frame,rvec,tvec,K,dist_coeffs,image_points)
            
        else:
            print("Cant solve PnP")
    except Exception as e:
        print(e)

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