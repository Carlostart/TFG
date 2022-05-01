import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import dataProcessing as dp

DEBUG = True


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
        # Separa y recorta cada moneda
        images = self.cropCircle(img, ncoins)
        # self.getLines(images)
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
        for image in images if ncoins > 1 else images[:1]:
            ocr_data = self.getOCR(image)   # Lectura de caracteres
            # Ajustamos el tamaño
            resized = cv2.resize(image, (dp.IM_SIZE_ADJ, dp.IM_SIZE_ADJ))
            Hu = self.huMoments(resized)      # Obtenemos los momentos de Hu
            keyP = self.keyPoints(resized)    # Obtenemos las esquinas
            # self.getAreas(resized)  # Detectamos formas y sus areas

            # Guardamos los datos obtenidos en el diccionario
            data.get("HU_1").append(Hu[0])
            data.get("HU_2").append(Hu[1])
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

    @classmethod
    def cropCircle(self, img,  ncoins=1):
        height, width = img.shape
        # Aplicamos un suavizado para eliminar ruidos
        aux = cv2.GaussianBlur(
            img, dp.HCIRCLES_GAUSS_KERNEL, dp.HCIRCLES_GAUSS_SIGMA)
        # Obtenemos los bordes de la imagen
        # (Multiplicamos la imagen por dos para aumentar las diferencias de intensidades)
        edges = cv2.Canny(aux*2, dp.CANNY_TRHES1, dp.CANNY_TRHES2)

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
        # mediana = np.median(circles[0, :, 2])
        # print(f"Mediana: {mediana}")

        # edges_ls = []
        cropped_ls = []
        # Pasamos a escala BGR para pocer marcar la imagen con colores
        show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
            cv2.circle(mask, (x, y), r, 255,
                       thickness=-1)
            # Aplicamos la máscara
            masked_data = cv2.bitwise_and(img, img, mask=mask)

            # Ajustamos la imagen al circulo
            cropped = masked_data[y-r:y+r, x-r:x+r]
            # cropped_edges = edges[y-r:y+r, x-r:x+r]
            # Añadimos a la lista de monedas en la imagen
            cropped_ls.append(cropped)
            # edges_ls.append(cropped_edges)

            # -- DEBUG} --
            #  Dibujamos los circulos seleccionados
            cv2.circle(show,  (x, y), r, (0, 255, 0),
                       thickness=2)

            if DEBUG:
                self.DCircles = show

            #  ------------

            # Nos salimos del bucle cuando hemos seleccionado el numero de monedas especificado
            if idx == ncoins - 1:
                break

        # return cropped_ls, edges_ls
        return cropped_ls

    def getLines(imgs):

        for img in imgs:
            # Aplicamos un suavizado para eliminar ruidos
            aux = cv2.GaussianBlur(
                img, dp.HLINES_GAUSS_KERNEL, dp.HLINES_GAUSS_SIGMA)
            # Obtenemos los bordes de la imagen
            # (Multiplicamos la imagen por dos para aumentar las diferencias de intensidades)
            edges = cv2.Canny(aux*4, dp.CANNY_TRHES1, dp.CANNY_TRHES2)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
            print(f"n lineas: {len(lines)}")
            print(f"lineas->{lines}")
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            for l1 in lines[:5]:
                r1, theta1 = l1[0]
                a, b = np.cos(theta1), np.sin(theta1)
                x0, y0 = a*r1, b*r1
                x1, y1, x2, y2 = int(
                    x0 + 1000*(-b)), int(y0 + 1000*(a)), int(x0 - 1000*(-b)), int(y0 - 1000*(a))
                cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if DEBUG:
                plt.subplot(248)
                plt.imshow(edges)
                plt.title('Lines')

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
            if tupla[-1] > dp.OCR_MINRATE:
                textos.append(tupla[1])

        if DEBUG:
            plt.subplot(245)
            plt.imshow(copy)
            plt.title("OCR")

        return textos

    @staticmethod
    def huMoments(img):
        # Calculamos los momentos estadisticos hasta los de primera orden
        M = cv2.moments(img, False)
        # Calculamos los momentos de Hu y nos quedamos con los dos primeros
        Hm = cv2.HuMoments(M).flatten()[0:dp.N_HUMOMS]
        return Hm

    @staticmethod
    def keyPoints(img):
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

    @staticmethod
    def getAreas(img):
        img = img.copy()
        height, width = img.shape

        # Aplicamos un suavizado para eliminar ruidos
        aux = cv2.GaussianBlur(
            img, dp.HLINES_GAUSS_KERNEL, dp.HLINES_GAUSS_SIGMA)
        # Obtenemos los bordes de la imagen
        # (Multiplicamos la imagen por dos para aumentar las diferencias de intensidades)
        edges = cv2.Canny(aux, dp.CANNY_TRHES1, dp.CANNY_TRHES2)
        # Ahora eliminamos el anillo exterior de la moneda
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (int(height/2), int(width/2)), int((height/2) * .7), 255,
                   thickness=-1)
        masked_data = cv2.bitwise_and(edges, edges, mask=mask)

        # Creamos un kernel para procesar la imagen
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        morphed = cv2.morphologyEx(
            masked_data, cv2.MORPH_CLOSE, kernel, iterations=1)
        plt.subplot(245)
        plt.imshow(morphed, 'gray')
        plt.title("CLOSE")
        # using a findContours() function
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        i = 0

        print(len(contours))
        # list for storing names of shapes
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for contour in contours:

            # here we are ignoring first counter because
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue

            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)

            # using drawContours() function
            cv2.drawContours(img, [contour], 0, (0, 0, 255), cv2.FILLED)

            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

        if DEBUG:
            plt.subplot(247)
            plt.imshow(masked_data, 'gray')
            plt.title('Edges')
            plt.subplot(248)
            plt.imshow(img)
            plt.title('Contours')
