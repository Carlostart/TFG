import scipy.ndimage as ndi
from ast import Try
import dataProcessing as dp
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr

DEBUG = True


class ImProcessing:

    def quickShow(img, title=""):
        plt.imshow(img, 'gray')
        plt.title(title)
        plt.show()

    @ classmethod
    def extractData(self, img_path: str, ncoins=1) -> dict[str, list]:

        print(f"Identificando imagen: {img_path}")
        img = cv2.imread(img_path)  # Lee imagen
        # -- Ajustes de imagen para que sigan un mismo formato --
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
        img = cv2.normalize(img, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX)
        # Separa y recorta cada moneda
        images = self.cropCircle(img, ncoins)

        # Diccionario de listas de datos para exportar al archivo csv
        data = dp.initData()

        # Obtenemos el nombre del archivo
        img_file = img_path.split('\\')[-1]
        # Si hay multiples monedas en la imagen debe estar indicado con M
        if img_file[0] == 'M':
            id = int(img_file.split('_')[1])
        else:
            id = int(img_file.split('_')[0])

        # Si hay varias monedas en una misma imagen procesamos todas
        for image in images:
            equalized = self.clahe(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            ocr_data = self.getOCR(equalized)   # Lectura de caracteres
            # Ajustamos el tamaño
            resized = cv2.resize(image, (dp.IM_SIZE_ADJ, dp.IM_SIZE_ADJ))
            Hu = self.huMoments(resized)      # Obtenemos los momentos de Hu
            # keyP = self.keyPoints(resized)    # Obtenemos las esquinas
            edges = self.edgesInside(image)
            # Calculamos centro de gravedad y orientamos imgaen
            rotated, cog = self.normalizeOrientation(edges, edges)
            self.getLines(rotated)

            # Guardamos los datos obtenidos en el diccionario
            data.get("HU_1").append(Hu[0])
            data.get("HU_2").append(Hu[1])
            data.get("CG_X").append(cog[0])
            data.get("CG_Y").append(cog[1])
            dp.appendOcrData(ocr_data, data)
            data.get("CLASS").append(id)

            if DEBUG:
                plt.subplot(241)
                plt.title("Original")
                plt.imshow(img, 'gray')

                plt.subplot(242)
                plt.title("Cropped")
                plt.imshow(image, 'gray')

                plt.subplot(243)
                plt.title("Canny")
                plt.imshow(self.DCanny, 'gray')

                plt.subplot(244)
                plt.title("Circles")
                plt.imshow(self.DCircles)

                plt.show()

        return data

    def removeExternalRing(img: cv2.Mat, pr) -> cv2.Mat:
        height, width = img.shape
        # Ahora eliminamos el anillo exterior de la moneda
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (int(height/2), int(width/2)), int((height/2) * pr), 255,
                   thickness=-1)
        return cv2.bitwise_and(img, img, mask=mask)

    @ staticmethod
    def clahe(img: cv2.Mat) -> cv2.Mat:
        # Cambiamos la imagen al modelo de color LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Los separamos en sus tres canales
        l, a, b = cv2.split(lab)
        # Aplicamos CLAHE (Contrast Limited Adaptive Histogram Equalization) al canal L
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
        cl = clahe.apply(l)
        # Y lo introducimos a la imagen
        limg = cv2.merge((cl, a, b))
        # Transformamos la imagen a escala de grises
        return cv2.cvtColor(cv2.cvtColor(
            limg, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)

    @ classmethod
    def cropCircle(self, img: cv2.Mat,  ncoins=1) -> list[cv2.Mat]:
        height, width = img.shape
        # Aplicamos un suavizado para eliminar ruidos
        blurred = cv2.GaussianBlur(
            img, dp.HCIRCLES_GAUSS_KERNEL, dp.HCIRCLES_GAUSS_SIGMA)
        # Obtenemos los bordes de la imagen
        # (Multiplicamos la imagen por dos para aumentar las diferencias de intensidades)
        edges = cv2.Canny(blurred*2, dp.CANNY_TRHES1,
                          dp.CANNY_TRHES2)

        # -- {DEBUG} --
        if DEBUG:
            self.DCanny = edges
        # -------------
        #   Obtenemos los circulos
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp.HCIRCLES_DP, minDist=dp.IM_SIZE_ADJ,
                                   param1=dp.HCIRCLES_PAR1, param2=dp.HCIRCLES_PAR2, minRadius=dp.HCIRCLES_MINRAD, maxRadius=0)

        # Muestra cuantos circulos se han encontrado
        print(f"Detectados {len(circles[0,:])} Circulos")
        # Obtenemos la mediana de los radios
        mediana = np.median(circles[0, :ncoins, 2])

        print(f"Mediana: {mediana}" if ncoins > 1 else '')

        cropped_ls = []
        # Pasamos a escala BGR para pocer marcar la imagen con colores
        show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Por cada circulo:
        miss = False
        deleted = 0
        for idx, i in enumerate(circles[0, :]):
            # -- Creamos una máscara --
            mask = np.zeros((height, width), np.uint8)
            x, y, r = map(int, i)  # Clasificamos centro y radio
            # Si el circulo se sale de la imagen reducimos el radio
            r = int(r*0.9)
            r = min(x, y, width-x, height-y, r)

            # Borra circulos de tamaño desproporcionado
            if (r > mediana*1.2 or r < mediana*0.8):
                print(f'Borra circulo: x={x} y={y} r={r}')
                miss = True
                deleted += 1
                continue

            # Dibujamos el circulo en la máscara
            cv2.circle(mask, (x, y), r, 255,
                       thickness=-1)
            # Aplicamos la máscara
            masked_data = cv2.bitwise_and(img, img, mask=mask)
            # Ajustamos la imagen al circulo
            cropped = masked_data[y-r:y+r, x-r:x+r]
            # Añadimos a la lista de monedas en la imagen
            cropped_ls.append(cropped)

            # -- DEBUG} --
            #  Dibujamos los circulos seleccionados
            cv2.circle(show,  (x, y), r, (0, 255, 0),
                       thickness=2)
            if DEBUG:
                self.DCircles = show
            #  ------------
            # Nos salimos del bucle cuando hemos seleccionado el numero de monedas especificado
            if idx == ncoins - 1 and (deleted > 3 or not miss):
                break
        return cropped_ls

    @ staticmethod
    def getOCR(img: cv2.Mat) -> list[str]:
        # Creamos el reader
        reader = easyocr.Reader(['en'])
        # Leemos la moneda
        output = reader.readtext(img)

        textos = []
        copy = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        for tupla in output:
            if tupla[-1] > dp.OCR_MINRATE:
                cord = tupla[0]

                # -- DEBUG --
                #  Muestra  donde detecto los caracteres
                x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
                x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
                cv2.rectangle(copy, (x_min, y_min),
                              (x_max, y_max), (255, 0, 0), 2)
                # -------------

                # Si el indice de acierto es > 70%
                textos.append(tupla[1])

        if DEBUG:
            plt.subplot(245)
            plt.imshow(copy)
            plt.title("OCR")

        print(f"Palabras encontradas -> {textos}")
        return textos

    @ staticmethod
    def huMoments(img: cv2.Mat) -> list:
        # Calculamos los momentos estadisticos hasta los de primera orden
        M = cv2.moments(img, False)
        # Calculamos los momentos de Hu y nos quedamos con los dos primeros
        Hm = cv2.HuMoments(M).flatten()[0:dp.N_HUMOMS]
        return Hm

    @ staticmethod
    def keyPoints(img: cv2.Mat):
        # Detección de esquinas
        dst1 = cv2.goodFeaturesToTrack(np.float32(
            img), dp.KP_MAXCORNERS, dp.KP_QUALITY, dp.KP_MINDIST)
        # Marcamos los puntos en la imagen para mostrarla
        kp = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        for corner in dst1:
            x, y = corner.ravel()
            cv2.circle(kp, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)

        if DEBUG:
            plt.subplot(246)
            plt.imshow(kp, 'gray')
            plt.title('KeyPoints')

        return dst1

    @ classmethod
    def edgesInside(self, img):
        # Normalizamos
        img = self.clahe(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        # Aplicamos un suavizado para eliminar ruidos
        aux = cv2.GaussianBlur(
            img, dp.HLINES_GAUSS_KERNEL, dp.HLINES_GAUSS_SIGMA)
        # Obtenemos los bordes de la imagen
        # (Multiplicamos la imagen por dos para aumentar las diferencias de intensidades)
        return cv2.Canny(aux, dp.CANNY_TRHES1, dp.CANNY_TRHES2)

    @ classmethod
    def getLines(self, edges: cv2.Mat):
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        if lines is not None:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            m = min(len(lines), dp.NUM_LINES)
            for line in lines[:m]:
                r, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a*r, b*r
                x1, y1, x2, y2 = int(
                    x0 + 1000*(-b)), int(y0 + 1000*(a)), int(x0 - 1000*(-b)), int(y0 - 1000*(a))
                cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if DEBUG:
                plt.subplot(248)
                plt.imshow(edges)
                plt.title('Lines')

            return lines[:m]
        else:
            return []

    @ classmethod
    def normalizeOrientation(self, edges: cv2.Mat, image: cv2.Mat):
        h, w = edges.shape
        cy, cx = h/2, w/2

        cmy, cmx = ndi.center_of_mass(edges)

        angle = math.degrees(math.atan2(cy-cmy, cx-cmx))

        M = cv2.getRotationMatrix2D((cx, cy), angle+180, 1)
        rotated = cv2.warpAffine(image, M, (w, h))

        if DEBUG:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            plt.subplot(246)
            cv2.circle(edges, (int(cx), int(cy)),
                       3, (255, 0, 0), cv2.FILLED)
            cv2.circle(edges, (int(cmx), int(cmy)),
                       5, (0, 255, 0), cv2.FILLED)
            plt.imshow(edges)
            plt.title('Centers')

            plt.subplot(247)
            plt.imshow(rotated, 'gray')
            plt.title("Rotated")

        return rotated, (cmx, cmy)
