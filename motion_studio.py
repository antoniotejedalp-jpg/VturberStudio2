"""Motion Studio Mini - seguimiento visual de movimiento con la webcam.

Requisitos: Python 3.11 o 3.12, opencv-python, mediapipe==0.10.21.
No utiliza avatar, voz ni grabacion de video: solo muestra el seguimiento en vivo.
"""

from collections import deque
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np


# Colores BGR para OpenCV.
FONDO = (29, 26, 23)
PANEL = (43, 38, 33)
BLANCO = (242, 238, 231)
GRIS = (176, 169, 158)
VERDE = (151, 214, 104)
CIAN = (242, 209, 109)
NARANJA = (93, 159, 240)
ROJO = (117, 113, 241)

ANCHO_CAMARA = 800
ALTO_CAMARA = 450
ANCHO_PANEL = 310
ALTO_ENCABEZADO = 72
ALTO_PIE = 68

# Conexiones simplificadas: hombros, brazos y torso.
CONEXIONES_TORSO = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]
PUNTOS_TORSO = [11, 12, 13, 14, 15, 16, 23, 24]

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def texto(lienzo, cadena, x, y, escala=0.58, color=BLANCO, grosor=1):
    cv2.putText(lienzo, cadena, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, grosor, cv2.LINE_AA)


def dibujar_etiqueta(lienzo, nombre, x, y, activo):
    color = VERDE if activo else GRIS
    cv2.circle(lienzo, (x, y), 5, color, -1, cv2.LINE_AA)
    texto(lienzo, nombre, x + 15, y + 5, 0.57, BLANCO if activo else GRIS)


def punto_en_imagen(landmark, ancho, alto):
    return (int(landmark.x * ancho), int(landmark.y * alto))


def dibujar_cuerpo(imagen, pose):
    if not pose:
        return
    alto, ancho = imagen.shape[:2]
    landmarks = pose.landmark

    def visible(indice):
        return landmarks[indice].visibility > 0.48

    for inicio, fin in CONEXIONES_TORSO:
        if visible(inicio) and visible(fin):
            cv2.line(imagen,
                     punto_en_imagen(landmarks[inicio], ancho, alto),
                     punto_en_imagen(landmarks[fin], ancho, alto),
                     CIAN, 3, cv2.LINE_AA)
    for indice in PUNTOS_TORSO:
        if visible(indice):
            cv2.circle(imagen, punto_en_imagen(landmarks[indice], ancho, alto),
                       5, BLANCO, -1, cv2.LINE_AA)


def dibujar_rastro(imagen, puntos, color):
    for indice in range(1, len(puntos)):
        anterior = puntos[indice - 1]
        actual = puntos[indice]
        grosor = max(1, int(5 * indice / max(1, len(puntos))))
        cv2.line(imagen, anterior, actual, color, grosor, cv2.LINE_AA)


def abrir_camara():
    # DirectShow suele evitar retrasos de inicializacion en Windows.
    if sys.platform.startswith("win"):
        camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if camara.isOpened():
            return camara
        camara.release()
    return cv2.VideoCapture(0)


def main():
    camara = abrir_camara()
    if not camara.isOpened():
        print("\nNo se pudo abrir la camara.")
        print("Comprueba los permisos de Windows, cierra otras aplicaciones y vuelve a intentar.")
        print("Si tienes otra webcam, cambia VideoCapture(0) por VideoCapture(1).")
        return 1

    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mostrar_rostro = True
    mostrar_manos = True
    mostrar_cuerpo = True
    mostrar_rastro = True
    rastros = {"izquierda": deque(maxlen=22), "derecha": deque(maxlen=22)}
    movimiento_anterior = None
    actividad = 0.0
    fps = 0.0
    instante_anterior = time.perf_counter()
    contador_capturas = 0

    nombre_ventana = "Motion Studio Mini - Captura de movimiento"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, ANCHO_CAMARA + ANCHO_PANEL,
                     ALTO_CAMARA + ALTO_ENCABEZADO + ALTO_PIE)

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as detector:
            while True:
                correcto, frame = camara.read()
                if not correcto:
                    print("No se pudo leer un fotograma de la camara.")
                    break

                # Vista espejo para facilitar seguir los propios movimientos.
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (ANCHO_CAMARA, ALTO_CAMARA))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                resultado = detector.process(frame_rgb)
                frame_rgb.flags.writeable = True

                hay_rostro = resultado.face_landmarks is not None
                hay_izquierda = resultado.left_hand_landmarks is not None
                hay_derecha = resultado.right_hand_landmarks is not None
                hay_manos = hay_izquierda or hay_derecha
                hay_cuerpo = resultado.pose_landmarks is not None

                if mostrar_rostro and hay_rostro:
                    mp_drawing.draw_landmarks(
                        frame, resultado.face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=VERDE, thickness=1, circle_radius=1),
                    )

                if mostrar_cuerpo and hay_cuerpo:
                    dibujar_cuerpo(frame, resultado.pose_landmarks)

                if mostrar_manos:
                    for mano, color in ((resultado.left_hand_landmarks, CIAN),
                                        (resultado.right_hand_landmarks, NARANJA)):
                        if mano is not None:
                            mp_drawing.draw_landmarks(
                                frame, mano, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=BLANCO, thickness=1, circle_radius=3),
                                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                            )

                # Trail de munecas y medidor simple de movimiento.
                actuales = {}
                for nombre, mano in (("izquierda", resultado.left_hand_landmarks),
                                     ("derecha", resultado.right_hand_landmarks)):
                    if mano is not None:
                        punto = punto_en_imagen(mano.landmark[0], ANCHO_CAMARA, ALTO_CAMARA)
                        actuales[nombre] = punto
                        rastros[nombre].append(punto)
                    else:
                        rastros[nombre].clear()

                if mostrar_rastro and mostrar_manos:
                    dibujar_rastro(frame, rastros["izquierda"], CIAN)
                    dibujar_rastro(frame, rastros["derecha"], NARANJA)

                # Estimacion visual (no medicion fisica) del movimiento.
                if hay_cuerpo:
                    lista = resultado.pose_landmarks.landmark
                    centro = (
                        (lista[11].x + lista[12].x) / 2,
                        (lista[11].y + lista[12].y) / 2,
                    )
                    actuales["centro"] = (int(centro[0] * ANCHO_CAMARA),
                                           int(centro[1] * ALTO_CAMARA))

                if movimiento_anterior is not None:
                    desplazamientos = []
                    for clave, posicion in actuales.items():
                        if clave in movimiento_anterior:
                            previo = movimiento_anterior[clave]
                            desplazamientos.append(np.hypot(posicion[0] - previo[0],
                                                            posicion[1] - previo[1]))
                    velocidad = float(np.mean(desplazamientos)) if desplazamientos else 0.0
                    actividad = 0.65 * actividad + 0.35 * velocidad
                movimiento_anterior = actuales

                ahora = time.perf_counter()
                delta = max(0.001, ahora - instante_anterior)
                fps = 0.85 * fps + 0.15 * (1 / delta)
                instante_anterior = ahora

                alto_total = ALTO_ENCABEZADO + ALTO_CAMARA + ALTO_PIE
                ancho_total = ANCHO_CAMARA + ANCHO_PANEL
                lienzo = np.full((alto_total, ancho_total, 3), FONDO, dtype=np.uint8)
                lienzo[ALTO_ENCABEZADO:ALTO_ENCABEZADO + ALTO_CAMARA, :ANCHO_CAMARA] = frame
                lienzo[:, ANCHO_CAMARA:] = PANEL

                texto(lienzo, "MOTION STUDIO", 25, 35, 0.85, BLANCO, 2)
                texto(lienzo, "CAPTURA DE MOVIMIENTO", 26, 57, 0.41, GRIS)
                texto(lienzo, f"{fps:.0f} FPS", ANCHO_CAMARA - 103, 40, 0.60, VERDE)

                panel_x = ANCHO_CAMARA + 24
                texto(lienzo, "SEGUIMIENTO EN VIVO", panel_x, 112, 0.62, BLANCO, 2)
                cv2.line(lienzo, (panel_x, 127), (ancho_total - 25, 127), GRIS, 1)
                dibujar_etiqueta(lienzo, "Rostro", panel_x + 7, 164, hay_rostro)
                dibujar_etiqueta(lienzo, "Manos", panel_x + 7, 202, hay_manos)
                dibujar_etiqueta(lienzo, "Torso y brazos", panel_x + 7, 240, hay_cuerpo)
                texto(lienzo, "MOVIMIENTO", panel_x, 293, 0.55, GRIS)
                estado = "ACTIVO" if actividad > 2.2 else "SUAVE / QUIETO"
                texto(lienzo, estado, panel_x, 321, 0.71,
                      VERDE if actividad > 2.2 else BLANCO, 2)
                #texto(lienzo, "La barra muestra una estimacion", panel_x, 359, 0.42, GRIS)
                #texto(lienzo, "relativa del desplazamiento.", panel_x, 379, 0.42, GRIS)
                cv2.rectangle(lienzo, (panel_x, 395), (ancho_total - 25, 413), FONDO, -1)
                longitud = min(ANCHO_PANEL - 49, int(actividad * 13))
                if longitud > 0:
                    cv2.rectangle(lienzo, (panel_x, 395), (panel_x + longitud, 413), VERDE, -1)
                #texto(lienzo, "Sin grabacion ni avatar", panel_x, 464, 0.46, GRIS)

                inferior = ALTO_ENCABEZADO + ALTO_CAMARA
                texto(lienzo, "1 Rostro   2 Manos   3 Cuerpo   4 Rastro", 25, inferior + 25, 0.57, BLANCO)
                texto(lienzo, "C Captura de pantalla     Q Salir", 25, inferior + 53, 0.55, GRIS)
                cv2.imshow(nombre_ventana, lienzo)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), ord("Q"), 27):
                    break
                if tecla == ord("1"):
                    mostrar_rostro = not mostrar_rostro
                if tecla == ord("2"):
                    mostrar_manos = not mostrar_manos
                if tecla == ord("3"):
                    mostrar_cuerpo = not mostrar_cuerpo
                if tecla == ord("4"):
                    mostrar_rastro = not mostrar_rastro
                if tecla in (ord("c"), ord("C")):
                    os.makedirs("capturas", exist_ok=True)
                    contador_capturas += 1
                    nombre = f"capturas/movimiento_{time.strftime('%Y%m%d_%H%M%S')}_{contador_capturas:02d}.png"
                    if cv2.imwrite(nombre, lienzo):
                        print(f"Captura guardada: {nombre}")
                    else:
                        print("No se pudo guardar la captura.")
    finally:
        camara.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
