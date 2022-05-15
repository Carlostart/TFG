import data_processing as dp

import scipy.ndimage as ndi
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr

DEBUG = False


class ImProcessing:
    @staticmethod
    def quickShow(img, title=""):
        plt.imshow(img, "gray")
        plt.title(title)
        plt.show()

    @staticmethod
    def removeExternalRing(img: cv2.Mat, pr) -> cv2.Mat:
        height, width = img.shape[:2]
        # Ahora eliminamos el anillo exterior de la moneda
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(
            mask,
            (int(height / 2), int(width / 2)),
            int((height / 2) * pr),
            255,
            thickness=-1,
        )
        return cv2.bitwise_and(img, img, mask=mask)

    @staticmethod
    def getOuterRing(img: cv2.Mat, pr) -> cv2.Mat:
        height, width = img.shape[:2]
        mask = np.ones((height, width), np.uint8)
        cv2.circle(
            mask,
            (int(height / 2), int(width / 2)),
            int((height / 2) * pr),
            0,
            thickness=-1,
        )
        return cv2.bitwise_and(img, img, mask=mask)

    @staticmethod
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
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def gaussBlur(img):
        h, _ = img.shape[:2]
        # Aplicamos un suavizado para eliminar ruidos
        ksize = int(h / dp.HLINES_KERNEL_RATIO)
        fm = cv2.Laplacian(img, cv2.CV_64F).var()
        if fm < 600:
            ksize = int(ksize * 0.7)
        ksize = ksize if ksize % 2 else ksize - 1
        sigma = (ksize - 1) / 6

        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        return blurred

    @classmethod
    def edgesInside(cls, img):
        aux = img
        for _ in range(1):
            # Aplicamos un suavizado para eliminar ruidos
            aux = cls.gaussBlur(aux)
            aux = cls.clahe(aux)
        # Obtenemos los bordes de la imagen
        edges = cv2.Canny(aux, dp.CANNY_TRHES1, dp.CANNY_TRHES2)

        return edges

    # https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df
    @staticmethod
    def correction(
        img,
        shadow_amount_percent,
        shadow_tone_percent,
        shadow_radius,
        highlight_amount_percent,
        highlight_tone_percent,
        highlight_radius,
        color_percent,
    ):

        shadow_tone = shadow_tone_percent * 255
        highlight_tone = 255 - highlight_tone_percent * 255

        shadow_gain = 1 + shadow_amount_percent * 6
        highlight_gain = 1 + highlight_amount_percent * 6

        # Separamos los canales RGB
        height, width = img.shape[:2]
        img = img.astype(np.float)
        img_R, img_G, img_B = (
            img[..., 2].reshape(-1),
            img[..., 1].reshape(-1),
            img[..., 0].reshape(-1),
        )

        # Todo el proceso de correccion esta hecho en YUV,
        # ajusta los hightlights y sombras en Y, los colores en UV,
        img_Y = 0.3 * img_R + 0.59 * img_G + 0.11 * img_B
        img_U = -img_R * 0.168736 - img_G * 0.331264 + img_B * 0.5
        img_V = img_R * 0.5 - img_G * 0.418688 - img_B * 0.081312

        # Extrae sombras y hightlights
        shadow_map = 255 - img_Y * 255 / shadow_tone
        shadow_map[np.where(img_Y >= shadow_tone)] = 0
        highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
        highlight_map[np.where(img_Y <= highlight_tone)] = 0

        # Suavizado de Gauss para transicion mas suave
        if shadow_amount_percent * shadow_radius > 0:
            shadow_map = cv2.GaussianBlur(
                shadow_map.reshape(height, width),
                ksize=(shadow_radius, shadow_radius),
                sigmaX=0,
            ).reshape(-1)

        if highlight_amount_percent * highlight_radius > 0:
            highlight_map = cv2.GaussianBlur(
                highlight_map.reshape(height, width),
                ksize=(highlight_radius, highlight_radius),
                sigmaX=0,
            ).reshape(-1)

        # Creamos LUT
        t = np.arange(256)
        LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
        LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + 0.5)))
        LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
        LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + 0.5)))

        # Ajustamos el tono
        shadow_map = shadow_map * (1 / 255)
        highlight_map = highlight_map * (1 / 255)

        iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
        iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
        img_Y = iH

        # Ajustamos el color
        if color_percent != 0:
            # LUT de color
            if color_percent > 0:
                LUT = (1 - np.sqrt(np.arange(32768)) * (1 / 128)) * color_percent + 1
            else:
                LUT = np.sqrt(np.arange(32768)) * (1 / 128) * color_percent + 1

            # Ajustamos la saturacion de color
            color_gain = LUT[np.int_(img_U**2 + img_V**2 + 0.5)]
            w = 1 - np.minimum(2 - (shadow_map + highlight_map), 1)
            img_U = w * img_U + (1 - w) * img_U * color_gain
            img_V = w * img_V + (1 - w) * img_V * color_gain

        # Re convertimos a RGB
        output_R = np.int_(img_Y + 1.402 * img_V + 0.5)
        output_G = np.int_(img_Y - 0.34414 * img_U - 0.71414 * img_V + 0.5)
        output_B = np.int_(img_Y + 1.772 * img_U + 0.5)

        output = np.row_stack([output_B, output_G, output_R]).T.reshape(
            height, width, 3
        )
        output = output = np.minimum(np.maximum(output, 0), 255).astype(np.uint8)
        return output

    @classmethod
    def extractData(cls, img_path: str, ncoins=1) -> dict[str, list]:

        print(f"Identificando imagen: {img_path}")
        img = cv2.imread(img_path)[:, :, ::-1]  # Lee imagen
        # -- Ajustes de imagen para que sigan un mismo formato --
        if ncoins > 0:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
            img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # Separa y recorta cada moneda
            images = cls.cropCircle(img, ncoins)
        else:
            images = [img]

        # Diccionario de listas de datos para exportar al archivo csv
        data = dp.initData()
        # Obtenemos el nombre del archivo
        class_id, _ = dp.getClass(img_path)
        # Si hay varias monedas en una misma imagen procesamos todas
        for image in images:
            equalized = cls.clahe(image)
            ocr_data = cls.getOCR(equalized)  # Lectura de caracteres
            reduced = cls.removeExternalRing(image, 0.9)
            ring = cls.getOuterRing(reduced, 0.65)
            rings2compare = [
                cv2.imread(ring, 0) for ring in dp.getFilesInFolders([dp.RINGS_FOLDER])
            ]
            ring_similarities = cls.compareImgs(ring, rings2compare)
            keyP = cls.keyPoints(reduced)  # Obtenemos info delas esquinas
            # Ajustamos el tamaño
            resized = cv2.resize(reduced, (dp.IM_SIZE_ADJ, dp.IM_SIZE_ADJ))
            gray_rs = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            Hu = cls.huMoments(gray_rs)  # Obtenemos los momentos de Hu
            edges = cls.edgesInside(image)
            edges = cls.removeExternalRing(edges, 0.7)

            # Calculamos centro de gravedad y orientamos imgaen
            rotated, cog = cls.normalizeOrientation(edges, edges)
            lines = cls.getLines(rotated)
            lines_data = dp.getLinesData(lines, rotated.shape[0])

            # Guardamos los datos obtenidos en el diccionario
            data.get("HU_1").append(Hu[0])
            data.get("HU_2").append(Hu[1])
            data.get("CG_X").append(cog[0])
            data.get("CG_Y").append(cog[1])
            data.get("CG_DIST").append(cog[2])
            data.get("CG_ANGLE").append(cog[3])
            data.get("CKP_X").append(keyP[0])
            data.get("CKP_Y").append(keyP[1])
            data.get("CKP_DIST").append(keyP[2])
            data.get("CKP_ANGLE").append(keyP[3])
            for k in lines_data:
                data.get(k).append(lines_data[k])
            for k in ring_similarities:
                data.get(k).append(ring_similarities[k])

            dp.appendOcrData(ocr_data, data)
            data.get("CLASS").append(class_id)

            if DEBUG:
                plt.subplot(241)
                plt.title("Original")
                plt.imshow(img)

                plt.subplot(242)
                plt.title("Cropped")
                plt.imshow(image)

                if ncoins > 0:
                    # plt.subplot(243)
                    # plt.title("Canny")
                    # plt.imshow(self.DCanny, 'gray')

                    plt.subplot(243)
                    plt.title("Circles")
                    plt.imshow(cls.DCircles)

                plt.show()

        return data

    @classmethod
    def cropCircle(cls, img: cv2.Mat, ncoins=1) -> list[cv2.Mat]:

        _, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        if threshold[0, 0, 0] == 255:
            percent = 1
            img = cls.correction(
                img, percent, percent, 500, percent, percent, 500, percent
            )

        height, width = img.shape[:2]
        # Aplicamos un suavizado para eliminar ruidos
        blurred = cls.gaussBlur(img)

        # plt.imshow(blurred, 'gray')
        # plt.show()
        # Obtenemos los bordes de la imagen
        # (Multiplicamos la imagen por dos para aumentar las diferencias de intensidades)
        edges = cv2.Canny(blurred * 2, dp.CANNY_TRHES1, dp.CANNY_TRHES2)

        ksize = int(img.shape[0] / dp.HLINES_KERNEL_RATIO)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        morphed = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
        # kernel = cv2.getStructuringElement(
        #     cv2.MORPH_ELLIPSE, (int(ksize/2), int(ksize/2)))
        # morphed = cv2.morphologyEx(
        #     morphed, cv2.MORPH_OPEN, kernel, iterations=1)
        # -- {DEBUG} --
        # if DEBUG:
        #     self.DCanny = morphed
        # -------------
        #   Obtenemos los circulos
        cdp = int(height / 255)
        circles = cv2.HoughCircles(
            morphed,
            cv2.HOUGH_GRADIENT,
            cdp,
            minDist=dp.IM_SIZE_ADJ,
            param1=dp.HCIRCLES_PAR1,
            param2=dp.HCIRCLES_PAR2,
            minRadius=dp.HCIRCLES_MINRAD,
            maxRadius=2000,
        )

        # Muestra cuantos circulos se han encontrado
        print(f"Detectados {len(circles[0,:])} Circulos")
        # Obtenemos la mediana de los radios

        rcoin = circles[0, 0, 2]
        print(f"Coin Size -> {rcoin}")

        cropped_ls = []
        # Pasamos a escala BGR para pocer marcar la imagen con colores
        if DEBUG:
            show = img.copy()
        # Por cada circulo:
        deleted = 0
        for idx, i in enumerate(circles[0, :]):
            miss = False
            # -- Creamos una máscara --
            mask = np.zeros((height, width), np.uint8)
            x, y, r = map(int, i)  # Clasificamos centro y radio
            # Si el circulo se sale de la imagen reducimos el radio
            r = min(x, y, width - x, height - y, r)

            # Borra circulos de tamaño desproporcionado
            if r > rcoin * 1.2 or r < rcoin * 0.8:
                print(f"Borra circulo: x={x} y={y} r={r}")
                miss = True
                deleted += 1
                continue

            # Dibujamos el circulo en la máscara
            cv2.circle(mask, (x, y), r, 255, thickness=-1)
            # Aplicamos la máscara
            masked_data = cv2.bitwise_and(img, img, mask=mask)
            # Ajustamos la imagen al circulo
            cropped = masked_data[y - r : y + r, x - r : x + r]
            # Añadimos a la lista de monedas en la imagen
            cropped_ls.append(cropped)

            # -- DEBUG} --
            #  Dibujamos los circulos seleccionados
            if DEBUG:
                cv2.circle(show, (x, y), r, (255, 0, 0), thickness=10)
                cls.DCircles = show
            #  ------------
            # Nos salimos del bucle cuando hemos seleccionado el numero de monedas especificado
            if idx >= ncoins - 1 and (deleted > 3 or not miss):
                break
        # plt.imshow(show)
        # plt.show()
        return cropped_ls

    @staticmethod
    def getOCR(img: cv2.Mat) -> list[str]:
        # Creamos el reader
        reader = easyocr.Reader(["en"])
        # Leemos la moneda
        output = reader.readtext(img)

        textos = []
        copy = img.copy()
        for tupla in output:
            if tupla[-1] > dp.OCR_MINRATE:
                cord = tupla[0]

                # -- DEBUG --
                #  Muestra  donde detecto los caracteres
                x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
                x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
                cv2.rectangle(copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                # -------------

                # Si el indice de acierto es > 70%
                textos.append(tupla[1])

        if DEBUG:
            plt.subplot(244)
            plt.imshow(copy)
            plt.title("OCR")

        print(f"Palabras encontradas -> {textos}")
        return textos

    @staticmethod
    def huMoments(img: cv2.Mat) -> list:
        # Calculamos los momentos estadisticos hasta los de primera orden
        M = cv2.moments(img, False)
        # Calculamos los momentos de Hu y nos quedamos con los dos primeros
        Hm = cv2.HuMoments(M).flatten()[0 : dp.N_HUMOMS]
        return Hm

    @classmethod
    def keyPoints(cls, img: cv2.Mat):
        # Detección de esquinas
        kps_img = cv2.cvtColor(cls.gaussBlur(img), cv2.COLOR_BGR2GRAY)

        kps = cv2.goodFeaturesToTrack(
            np.float32(kps_img), dp.KP_MAXCORNERS, dp.KP_QUALITY, dp.KP_MINDIST
        )
        # Marcamos los puntos en la imagen para mostrarla
        for corner in kps:
            x, y = corner.ravel()
            cv2.circle(kps_img, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)

        x, y = np.mean(kps[:, 0, 0]), np.mean(kps[:, 0, 1])

        if DEBUG:
            plt.subplot(245)
            plt.imshow(kps_img)
            plt.title("KeyPoints")

        h = img.shape[0]
        c = int(h / 2)
        dist = np.sqrt((x - c) ** 2 + (y - c) ** 2) * dp.IM_SIZE_ADJ / h
        angle = math.degrees(math.atan2(y - c, x - c)) % 360
        return x, y, dist, angle

    @classmethod
    def getLines(cls, edges: cv2.Mat):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CROSS, kernel, iterations=1)

        lines = cv2.HoughLinesP(morphed, 1, np.pi / 360, 50)
        lines = np.concatenate(lines)
        if lines is not None:
            morphed = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)

            m = min(len(lines), dp.NUM_LINES)
            if DEBUG:
                sorted_lines = sorted(
                    lines,
                    key=lambda x: ((x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2) ** 0.5,
                    reverse=True,
                )

                for line in lines[:m]:
                    cv2.line(morphed, tuple(line[:2]), tuple(line[2:4]), (0, 255, 0), 2)

                for line in sorted_lines[:m]:
                    cv2.line(morphed, tuple(line[:2]), tuple(line[2:4]), (255, 0, 0), 2)

                plt.subplot(248)
                plt.imshow(morphed)
                plt.title("Lines")
            return lines
        else:
            return []

    @classmethod
    def normalizeOrientation(cls, edges: cv2.Mat, image: cv2.Mat):
        h, w = edges.shape
        cy, cx = h / 2, w / 2

        cmy, cmx = ndi.center_of_mass(edges)

        dist = np.sqrt((cx - cmx) ** 2 + (cy - cmy) ** 2) * dp.IM_SIZE_ADJ / h
        print(f"Center Dist -> {dist}")
        angle = math.degrees(math.atan2(cy - cmy, cx - cmx)) % 360
        if dist > dp.MIN_CENTERS_DIST:
            M = cv2.getRotationMatrix2D((cx, cy), angle + 180, 1)
            rotated = cv2.warpAffine(image, M, (w, h))
        else:
            rotated = image

        if DEBUG:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            plt.subplot(246)
            cv2.circle(edges, (int(cx), int(cy)), 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(edges, (int(cmx), int(cmy)), 5, (0, 255, 0), cv2.FILLED)
            cv2.line(edges, (int(cx), int(cy)), (int(cmx), int(cmy)), (0, 0, 255), 2)
            plt.imshow(edges)
            plt.title("Centers")

            plt.subplot(247)
            plt.imshow(rotated, "gray")
            plt.title("Rotated")

        return rotated, (cmx, cmy, dist, angle)

    @staticmethod
    def compareImgs(img1: cv2.Mat, img2_ls: list[cv2.Mat]):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        data = {}
        for i, img2 in enumerate(img2_ls):
            kp2, des2 = sift.detectAndCompute(img2, None)
            matches = flann.knnMatch(des1, des2, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < dp.SIFT_PERCENTAGE_FOR_GP * n.distance:
                    good_points.append(m)

            matching_result = cv2.drawMatches(
                img1, kp1, img2, kp2, good_points, None, flags=2
            )

            r = len(good_points) / min(len(kp1), len(kp2))
            data[f"RING_SIMILARITY_{i}"] = r

            plt.imshow(matching_result)
            plt.title("SIFT")
            plt.show()

        return data
