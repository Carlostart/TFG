import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr
import pandas
import csv

class ImProcessing:

    @classmethod
    def extractData(self, img_path, multiple=False):
        img = cv2.imread(img_path)
        # Ajustes de imagen para que sigan un mismo formato
        img = self.increaseContrast(img)
        images = self.cropCircle(img, multiple)

        data = []
        for image in images if multiple else images[:1]:

            ocr_data = self.getOCR(image)      # Lectura de caracteres
            Hu = self.huMoments(image)    # Obtenemos los momentos de Hu
            keyP = self.keyPoints(image)  # Obtenemos las esquinas
            # areas = self.getAreas(image)  # Detectamos formas y sus areas

            # Espeamos a pulsar una tecla para cerrar las imágenes
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            data.append((image, ocr_data, Hu, keyP))

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

        return img

    @staticmethod
    def cropCircle(img, multiple=False):
        # -- Primero modificamos la imagen para facilitar la detección de circulos --
        #   Creamos una copia de la imagen para poderla modificar
        gray = img.copy()
        #   Aplicamos un umbral
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # -- {DEBUG} --
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        # plt.imshow(thresh,'gray')
        # plt.show()

        #   Aseguramos que el fondo sea negro
        if thresh[0, 0] != 0:
            thresh = (255-thresh)

        #   Usaremos un kernel 5x5
        kernel = np.ones((5, 5), np.uint8)
        #   Aplicamos 5 dilataciones
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # -- {DEBUG} --
        # cv2.imshow("1", thresh)
        # cv2.waitKey(0)
        # plt.imshow(thresh,'gray')
        # plt.show()

        #   Aplicamos una erosion
        # thresh = cv2.erode(thresh, kernel)

        # cv2.imshow("2", thresh)
        # cv2.waitKey(0)

        #   Eliminamos artefactos y suavizamos superficie
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # -- {DEBUG} --
        # cv2.imshow('3', thresh)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        plt.subplot(221)
        plt.imshow(thresh,'gray')


        # Detección de circulos
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 4,
                                   255, param1=50, param2=250, minRadius=127, maxRadius=0)
        # Si no encuentra hace una busqueda con mayor indice de error
        if circles is None:
            circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 4,
                                       255, param1=50, param2=100, minRadius=127, maxRadius=0)
        print(circles)

        imgs = []
        height, width = img.shape
        gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        for i in circles[0, :]:
            # Eliminamos el anillo exterior de la moneda
            # -- Creamos una máscara --
            mask = np.zeros((height, width), np.uint8)
            # i[2] = i[2]*.75
            # Dibujamos los circulos en la máscara
            x, y, r = i
            x, y, r = int(x), int(y), int(r)
            r = min(x,y,width-x,height-y, r)

            cv2.circle(mask, (x,y), r, (255, 255, 255),
                       thickness=-1)

            # {DEBUG} Dibujamos los circulos seleccionados
            cv2.circle(gray,  (x,y), r, (0, 255, 0),
                       thickness=2)

            # Aplicamos la máscara
            masked_data = cv2.bitwise_and(img, img, mask=mask)

            # -- Ajustamos la imagen al circulo --
            cropped = masked_data[y-r:y+r, x-r:x+r]
            cropped = cv2.resize(cropped, (256, 256))
            imgs.append(cropped)

            # -- {DEBUG} --
            # cv2.imshow("final", cropped)
            # cv2.waitKey(0)
            plt.subplot(222)
            plt.imshow(gray)

            if not multiple:
                break

        # -- {DEBUG} --
        # cv2.imshow("circles", gray)
        # plt.imshow(gray)
        # plt.show()
        # cv2.destroyAllWindows()
        return imgs

    @staticmethod
    def getOCR(img):
        reader = easyocr.Reader(['en'])
        output = reader.readtext(img)

        textos = []

        # -- {DEBUG} Muestra  donde detecto los caracteres --
        copy = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
        for tupla in output:
            cord = tupla[0]

            x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
            x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
            cv2.rectangle(copy, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), 2)

            textos.append(tupla[1])

        # cv2.imshow("OCR_img", img)
        plt.subplot(223)
        plt.imshow(copy)
        return textos

    @staticmethod
    def huMoments(img):
        # Calculamos los momentos estadisticos hasta los de primera orden
        M = cv2.moments(img, False)
        # Calculamos los momentos de Hu y nos quedamos con los dos primeros
        Hm = cv2.HuMoments(M).flatten()[0:2]
        return Hm

    @staticmethod
    def keyPoints(img):
        # Detección de esquinas
        dst1 = cv2.goodFeaturesToTrack(np.float32(img), 25, 0.1, 10)
        # Marcamos los puntos en la imagen para mostrarla
        kp = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        for corner in dst1:
            x, y = corner.ravel()
            cv2.circle(kp, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)

        # cv2.imshow("keypoints mineigenvalues", kp)
        plt.subplot(224)
        plt.imshow(kp,'gray')
        plt.show()

        return dst1

    # @staticmethod
    # def getAreas(img):

    #     pass
