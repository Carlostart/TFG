import re
import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr
import pandas

IM_SIZE_ADJ = 255

GAUSS_KERNEL = (19, 19)
GAUSS_SIGMA = 5

CANNY_TRHES1 = 50
CANNY_TRHES2 = 100

HCIRCLES_DP = 4
HCIRCLES_PAR1 = 50
HCIRCLES_PAR2 = 100
HCIRCLES_MINRAD = 127

OCR_MINRATE = 0.7

N_HUMOMS = 2

KP_MAXCORNERS = 10
KP_QUALITY = 0.1
KP_MINDIST = 5


class ImProcessing:

    @classmethod
    def extractData(self, img_path, ncoins=1):
        print(f"Identificando imagen: {img_path}")
        img = cv2.imread(img_path)  # Lee imagen
        # -- Ajustes de imagen para que sigan un mismo formato --
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
        # Normalizamos
        img = cv2.normalize(img, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX)
        # self.increaseContrast(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR))
        images = self.cropCircle(img, ncoins)   # Separa y recorta cada moneda

        # Diccionario de listas de datos para exportar al archivo csv
        data = {"ID": [],
                "HU_1": [],
                "HU_2": [],
                "OCR_1": [],
                "OCR_2": [],
                "OCR_3": [],
                }

        img = img_path.split('/')[-1]
        # Si hay multiples monedas en la imagen debe estar indicado con M
        if img[0] == 'M':
            id = int(img.split('_')[1])
        else:
            id = int(img.split('_')[0])

        # Si hay varias monedas en una misma imagen procesamos todas
        for image in images if ncoins > 1 else images[:1]:
            # copy = image.copy()
            # copy = cv2.Canny(copy,50,100)
            ocr_data = self.getOCR(image)   # Lectura de caracteres
            # Ajustamos el tamaño
            resized = cv2.resize(image, (IM_SIZE_ADJ, IM_SIZE_ADJ))
            Hu = self.huMoments(resized)      # Obtenemos los momentos de Hu
            keyP = self.keyPoints(resized)    # Obtenemos las esquinas
            # areas = self.getAreas(image)  # Detectamos formas y sus areas

            # Guardamos los datos obtenidos en el diccionario
            data.get("ID").append(id)

            # Añade las tres primeras palabras (si las hay) detectadas por ocr
            nocr = 0
            for i, ocr in zip(range(3), ocr_data):
                data.get(f"OCR_{i+1}").append(ocr)
                nocr = i+1
            # Si no las marca como vacias
            for i in range(nocr, 3):
                data.get(f"OCR_{i+1}").append("")

            data.get("HU_1").append(Hu[0])
            data.get("HU_2").append(Hu[1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # plt.show()

        return data

    @staticmethod
    def increaseContrast(img):
        # Cambiamos la imagen al modelo de color LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Los separamos en sus tres canales
        l, a, b = cv2.split(lab)
        # Aplicamos CLAHE (Contrast Limited Adaptive Histogram Equalization) al canal L
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # Y lo introducimos a la imagen
        limg = cv2.merge((cl, a, b))
        # Transformamos la imagen a escala de grises
        img = cv2.cvtColor(cv2.cvtColor(
            limg, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)

    @staticmethod
    def cropCircle(img, ncoins=1):
        height, width = img.shape
        # -- Primero modificamos la imagen para facilitar la detección de circulos --
        #   Creamos una copia de la imagen para modificar
        gray = img.copy()
        #   Aplicamos un suavizado para eliminar ruidos
        gray = cv2.GaussianBlur(img, GAUSS_KERNEL, GAUSS_SIGMA)
        # Aumentamos las diferencias de intensidades para marcar mejor los bordes
        gray = gray*2

        edges = cv2.Canny(gray, CANNY_TRHES1, CANNY_TRHES2)

        # -- {DEBUG} --
        plt.subplot(241)
        plt.imshow(edges, 'gray')
        plt.title('Canny')
        # -------------

        #   Obtenemos los circulos
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, HCIRCLES_DP, minDist=IM_SIZE_ADJ,
                                   param1=HCIRCLES_PAR1, param2=HCIRCLES_PAR2, minRadius=HCIRCLES_MINRAD, maxRadius=0)

        # Muestra cuantos circulos se han encontrado
        print(f"Detectados {len(circles[0,:])} Circulos")

        # Obtenemos la mediana de los radios
        # mediana = np.median(circles[0, :, 2])
        # print(f"Mediana: {mediana}")

        imgs = []
        # Pasamos a escala BGR para pocer marcar la imagen con colores
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Por cada circulo:
        for idx, i in enumerate(circles[0, :]):
            # -- Creamos una máscara --
            mask = np.zeros((height, width), np.uint8)
            x, y, r = map(int, i)  # Clasificamos centro y radio
            # Si el circulo se sale de la imagen reducimos el radio
            r = min(x, y, width-x, height-y, r)

            # Borra circulos de tamaño desproporcionado
            # if (r > mediana*1.35 or r < mediana*0.35) and idx < len(circles[0, :]):
            #     # print(f'Borra circulo: x={x} y={y} r={r}')
            #     continue

            # Dibujamos el circulo en la máscara
            cv2.circle(mask, (x, y), r, (255, 255, 255),
                       thickness=-1)
            # Aplicamos la máscara
            masked_data = cv2.bitwise_and(img, img, mask=mask)

            # Ajustamos la imagen al circulo
            cropped = masked_data[y-r:y+r, x-r:x+r]
            # Añadimos a la lista de monedas en la imagen
            imgs.append(cropped)

            # -- DEBUG} --
            #  Dibujamos los circulos seleccionados
            cv2.circle(gray,  (x, y), r, (0, 255, 0),
                       thickness=2)
            plt.subplot(245)
            plt.title("Circles")
            plt.imshow(gray)
            #  ------------

            # Nos salimos del bucle cuando hemos seleccionado el numero de monedas especificado
            if idx == ncoins - 1:
                break

        return imgs

    @staticmethod
    def getOCR(img):
        # Creamos el reader
        reader = easyocr.Reader(['en'])
        # Leemos la moneda
        output = reader.readtext(img)

        textos = []
        copy = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        for tupla in output:
            cord = tupla[0]

            # -- DEBUG --
            #  Muestra  donde detecto los caracteres
            x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
            x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
            cv2.rectangle(copy, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), 2)
            # -------------

            # Si el indice de acierto es > 70%
            if tupla[-1] > OCR_MINRATE:
                textos.append(tupla[1])

        plt.subplot(246)
        plt.imshow(copy)

        return textos

    @staticmethod
    def huMoments(img):
        # Calculamos los momentos estadisticos hasta los de primera orden
        M = cv2.moments(img, False)
        # Calculamos los momentos de Hu y nos quedamos con los dos primeros
        Hm = cv2.HuMoments(M).flatten()[0:N_HUMOMS]
        return Hm

    @staticmethod
    def keyPoints(img):
        # Detección de esquinas
        dst1 = cv2.goodFeaturesToTrack(np.float32(
            img), KP_MAXCORNERS, KP_QUALITY, KP_MINDIST)
        # Marcamos los puntos en la imagen para mostrarla
        kp = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        for corner in dst1:
            x, y = corner.ravel()
            cv2.circle(kp, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)

        # cv2.imshow("keypoints mineigenvalues", kp)
        plt.subplot(247)
        plt.imshow(kp, 'gray')

        return dst1

    # @staticmethod
    # def getAreas(img):

    #     pass
