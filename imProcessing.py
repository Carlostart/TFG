import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr
import pandas

class ImProcessing:

    @classmethod
    def extractData(self, img_path, ncoins=1):
        print(f"Identificando imagen: {img_path}")
        img = cv2.imread(img_path)
        # -- Ajustes de imagen para que sigan un mismo formato --
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Grayscale
        # Normalizamos
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # self.increaseContrast(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR))

        images = self.cropCircle(img, ncoins)

        data = []
        # Si hay varias monedas en una misma imagen procesamos todas
        for image in images if ncoins > 1 else images[:1]:
            
            # copy = image.copy()
            # copy = cv2.Canny(copy,50,100)

            ocr_data = self.getOCR(image)      # Lectura de caracteres
            Hu = self.huMoments(image)    # Obtenemos los momentos de Hu
            keyP = self.keyPoints(image)  # Obtenemos las esquinas
            # areas = self.getAreas(image)  # Detectamos formas y sus areas

            # Espeamos a pulsar una tecla para cerrar las imágenes
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            plt.show()
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


    @staticmethod
    def cropCircle(img, ncoins=1):
        height, width = img.shape
        # -- Primero modificamos la imagen para facilitar la detección de circulos --
        #   Creamos una copia de la imagen para poderla modificar
        gray = img.copy()
        #   Aplicamos un umbral
        gray = cv2.GaussianBlur(img, (19, 19), 5)
        gray = gray*2
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
       
        edges = cv2.Canny(gray, 50, 100)
        # -- {DEBUG} --
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        plt.subplot(241)
        plt.imshow(edges,'gray')
        plt.title('Canny')
        #   Eliminamos artefactos y suavizamos superficie

        #   Aseguramos que el fondo sea negro
        if np.mean(thresh[0, :]) > 127:
            thresh = (255-thresh)
        plt.subplot(242)
        plt.imshow(thresh, 'gray')
        plt.title("Thresh")
        # plt.imshow(thresh,'gray')
        # plt.show()
        #   Usaremos un kernel 5x5
        kernel = np.ones((5, 5), np.uint8)
         
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations=1)
        plt.subplot(243)
        plt.imshow(morphed, 'gray')
        plt.title("Morphed")
        # morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel,iterations=1)
        # plt.subplot(223)
        # plt.imshow(morphed, 'gray')
        # plt.title("OPEN")
        # morphed = cv2.morphologyEx(morphed, cv2.MORPH_DILATE,kernel,iterations=5)
        # plt.subplot(224)
        # plt.imshow(morphed, 'gray')
        # plt.title("DILATE")

        contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)

        for row in range(height):
            if morphed[row, 0] == 255 and [0,row] not in big_contour:
                cv2.floodFill(morphed, None, (0, row), 0)
            if morphed[row, width-1] == 255 and [width-1,row] not in big_contour:
                cv2.floodFill(morphed, None, (width-1, row), 0)
        for col in range(width):
            if morphed[0, col] == 255 and [col,0] not in big_contour:
                cv2.floodFill(morphed, None, (col, 0), 0)
            if morphed[height-1, col] == 255 and [col,height-1] not in big_contour:
                cv2.floodFill(morphed, None, (col, height-1), 0)

        cv2.drawContours(morphed, [big_contour], 0, 255)

        # -- {DEBUG} --
        # cv2.imshow('3', morphed)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        plt.subplot(244)
        plt.imshow(morphed,'gray')
        plt.title("No borders shapes")
        # plt.show()
        print("Con canny")
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 4,
                        255, param1=50, param2=100, minRadius=127, maxRadius=0)
        # Detección de circulos
        if circles is None:
            print("NONE Morphed")
            circles = cv2.HoughCircles(morphed, cv2.HOUGH_GRADIENT, 4,
                                    255, param1=100, param2=250, minRadius=127, maxRadius=0)
        # Si no encuentra hace una busqueda con mayor indice de error
        if circles is None:
            print("NONE2  Aumenta param 2")
            circles = cv2.HoughCircles(morphed, cv2.HOUGH_GRADIENT, 4,
                                       255, param1=50, param2=100, minRadius=127, maxRadius=0)
        if circles is None:
            print("NON3 Canny Morphed")
            edges = cv2.Canny(morphed, 50, 100)
            plt.subplot(244)
            plt.imshow(edges,'gray')
            plt.title('Canny en morphed')
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 4,
                                    255, param1=50, param2=100, minRadius=127, maxRadius=0)

        print(f"Detectados {len(circles[0,:])} Circulos")
        if ncoins > 1:
            mediana = np.median(circles[0,:,2])
            print(f"Mediana: {mediana}")

        imgs = []
        gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        for idx, i in enumerate(circles[0, :]):
            # Eliminamos el anillo exterior de la moneda
            # -- Creamos una máscara --
            mask = np.zeros((height, width), np.uint8)
            # i[2] = i[2]*.75
            # Dibujamos los circulos en la máscara
            x, y, r = i
            x, y, r = int(x), int(y), int(r)
            r = min(x,y,width-x,height-y, r)

            # Borra circulos de tamaño desproporcionado
            if ncoins > 1 and (r > mediana*1.35 or r < mediana*0.35) and idx < len(circles[0,:]):
                # print(f'Borra circulo: x={x} y={y} r={r}')
                continue

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
            plt.subplot(245)
            plt.title("Circles")
            plt.imshow(gray)

            if idx == ncoins - 1:
                break

        # -- {DEBUG} --
        # cv2.imshow("circles", gray)
        # plt.imshow(gray)
        # plt.show()
        # cv2.destroyAllWindows()
        return imgs[:ncoins+1]

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
        plt.subplot(246)
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
        plt.subplot(247)
        plt.imshow(kp,'gray')

        return dst1

    # @staticmethod
    # def getAreas(img):

    #     pass
