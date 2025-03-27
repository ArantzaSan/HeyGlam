import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

imagen = cv2.imread("CARA.jpg")
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

rostros = detector(gris)

for rostro in rostros:
    puntos_faciales = predictor(gris, rostro)
    puntos = np.array([[puntos_faciales.part(i).x, puntos_faciales.part(i).y] for i in range(68)])

    # Medidas para proporciones faciales
    distancia_ojos = np.linalg.norm(puntos[36] - puntos[45])
    distancia_nariz_menton = np.linalg.norm(puntos[30] - puntos[8])
    ancho_cara = np.linalg.norm(puntos[0] - puntos[16])
    longitud_cara = np.linalg.norm(puntos[27] - puntos[8])
    ancho_nariz = np.linalg.norm(puntos[31] - puntos[35])
    longitud_nariz = np.linalg.norm(puntos[27] - puntos[33])
    longitud_labio_superior = np.linalg.norm(puntos[62] - puntos[66])
    longitud_labio_inferior = np.linalg.norm(puntos[66] - puntos[57])

    # Proporciones faciales
    proporcion_nariz_ojos = ancho_nariz / distancia_ojos
    proporcion_nariz_cara = longitud_nariz / longitud_cara
    proporcion_cara = longitud_cara / ancho_cara
    proporcion_labios = longitud_labio_inferior / longitud_labio_superior

    # Tercios faciales
    tercio_superior = np.linalg.norm(puntos[27] - puntos[19])
    tercio_medio = np.linalg.norm(puntos[27] - puntos[33])
    tercio_inferior = np.linalg.norm(puntos[33] - puntos[8])

    # Imprimir resultados e interpretaciones
    print("Proporción nariz/ojos:", proporcion_nariz_ojos)
    if proporcion_nariz_ojos < 0.7:
        print("   - Nariz relativamente estrecha.")
    elif proporcion_nariz_ojos > 1.0:
        print("   - Nariz relativamente ancha.")

    print("Proporción nariz/cara:", proporcion_nariz_cara)
    if proporcion_nariz_cara < 0.3:
        print("   - Nariz relativamente corta.")
    elif proporcion_nariz_cara > 0.45:
        print("   - Nariz relativamente larga.")

    print("Proporción cara:", proporcion_cara)
    if proporcion_cara < 1.3:
        print("   - Cara relativamente ancha.")
    elif proporcion_cara > 1.6:
        print("   - Cara relativamente alargada.")

    print("Proporción labios:", proporcion_labios)
    if proporcion_labios < 0.8:
        print("   - Labio superior más prominente.")
    elif proporcion_labios > 1.2:
        print("   - Labio inferior más prominente.")

    print("Tercios faciales (superior, medio, inferior):", tercio_superior, tercio_medio, tercio_inferior)
    if abs(tercio_superior - tercio_medio) < 10 and abs(tercio_medio - tercio_inferior) < 10:
        print("   - Tercios faciales relativamente equilibrados.")
    else:
        print("   - Tercios faciales desequilibrados.")

    # Dibujar puntos faciales
    for x, y in puntos:
        cv2.circle(imagen, (x, y), 2, (0, 255, 0), -1)

cv2.imshow("Rostros con puntos faciales", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()