import data_processing as dp

import scipy.ndimage as ndi
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr

from sewar.full_ref import (
    mse,
    rmse,
    psnr,
    uqi,
    ssim,
    ergas,
    scc,
    rase,
    sam,
    msssim,
    vifp,
)

EXTRACT_OCR = False
EXTRACT_RINGSIMS = False

NORMALIZE_ORIENTATION = False

EXTRACT_HU = True
EXTRACT_KEYPWR = True
EXTRACT_COG_GRAY = True
EXTRACT_COG_CANNY = True
EXTRACT_LINES = True
ESTRACT_PIXELS_DATA = True
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
        # print(f"FM-> {fm}")
        if fm < 600:
            ksize = int(ksize * 0.7)
        if fm < 300:
            ksize = int(ksize * 0.7)

        ksize = ksize if ksize % 2 else ksize - 1
        sigma = (ksize - 1) / 6

        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        return blurred

    @classmethod
    def edgesInside(cls, img):
        aux = img
        for _ in range(2):
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

        print(
            "==========================================================\n"
            + f"Identificando imagen: {img_path}"
        )
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
        all_data = dp.initData()
        # Obtenemos el nombre del archivo
        class_id, _ = dp.getClass(img_path)
        print(f"CLASS ID -> {class_id}")
        # Si hay varias monedas en una misma imagen procesamos todas
        rings2compare = [
            cv2.imread(ring) for ring in dp.getFilesInFolders([dp.RINGS_FOLDER])
        ]
        for image in images:
            data = {}
            # Aplicamos CLAHE y eliminamos el borde exterior de la moneda
            equalized = cls.clahe(image)
            reduced = cls.removeExternalRing(image, 0.9)
            img_without_ring = cls.removeExternalRing(image, 0.72)
            # Procesamos la imagen ajustando el tamaño, haciendo escala de grises y calculando bordes
            resized = cv2.resize(reduced, (dp.IM_SIZE_ADJ, dp.IM_SIZE_ADJ))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray_without_ring = cls.removeExternalRing(gray, 0.72)

            edges = cls.edgesInside(reduced)
            # Separamos el anillo de la moneda en diferentes formatos
            ring = cls.getOuterRing(reduced, 0.72)
            ring_edges = cls.getOuterRing(edges, 0.72)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_edges = cv2.morphologyEx(
                edges, cv2.MORPH_DILATE, kernel, iterations=1
            )
            edges_without_ring = cls.removeExternalRing(edges, 0.72)
            # Empezamos con la extracción de datos
            # Momentos de Hu
            if EXTRACT_HU:
                data_Hu = cls.huMoments(resized)
                print(f"Hu Moments ->", end=" ")
                for i in range(dp.N_HUMOMS):
                    print(data_Hu[i], end=" ")
                    data.update({f"HU_{i+1}": data_Hu[i]})
                print()
            # Lectura de caracteres
            if EXTRACT_OCR:
                data_ocr = cls.getOCR(equalized)
                dp.appendOcrData(data_ocr, data)
            # Obtenemos info delas esquinas
            if EXTRACT_KEYPWR:
                data_keyP = cls.keyPoints(reduced)  # WITH RING
                data_keyP_nr = cls.keyPoints(img_without_ring)  # WITH RING
                print(
                    f"KeyPs Center Info -> Dist ({round(data_keyP[3],2)}) Angle ({round(data_keyP[4],2)})"
                )
                data.update(
                    {
                        "CKP_N": data_keyP[0],
                        "CKP_X": data_keyP[1],
                        "CKP_Y": data_keyP[2],
                        "CKP_DIST": data_keyP[3],
                        "CKP_ANGLE": data_keyP[4],
                        "CKP_NR_N": data_keyP_nr[0],
                        "CKP_NR_X": data_keyP_nr[1],
                        "CKP_NR_Y": data_keyP_nr[2],
                        "CKP_NR_DIST": data_keyP_nr[3],
                        "CKP_NR_ANGLE": data_keyP_nr[4],
                    }
                )

                # Normalizamos la orientación de las imagenes que usaremos
                if NORMALIZE_ORIENTATION:
                    edges_without_ring = cls.normalize_orientation(
                        data_keyP[2], data_keyP[3], edges_without_ring
                    )
                    gray_without_ring = cls.normalize_orientation(
                        data_keyP[2], data_keyP[3], gray_without_ring
                    )
                    gray = cls.normalize_orientation(data_keyP[2], data_keyP[3], gray)
                    edges = cls.normalize_orientation(data_keyP[2], data_keyP[3], edges)

            # Obtenemos información de las líneas
            if EXTRACT_LINES:
                lines = cls.getLines(edges_without_ring)
                data_lines = dp.getLinesData(lines, edges_without_ring.shape[0])
                # print(f"Lines Info -> {data_lines}")
                data.update(data_lines)

            # Obtenemos centro de gravedad de la imagen en escala de grises
            if EXTRACT_COG_GRAY:
                data_cog_gray = cls.center_of_gravity_info(gray)
                data_cog_gray_nr = cls.center_of_gravity_info(gray_without_ring)
                print(
                    f"Gray COG Info -> Dist ({round(data_cog_gray[2],2)}) Angle ({round(data_cog_gray[3],2)})"
                )
                data.update(
                    {
                        "CGG_X": data_cog_gray[0],
                        "CGG_Y": data_cog_gray[1],
                        "CGG_DIST": data_cog_gray[2],
                        "CGG_ANGLE": data_cog_gray[3],
                        "CGG_NR_X": data_cog_gray_nr[0],
                        "CGG_NR_Y": data_cog_gray_nr[1],
                        "CGG_NR_DIST": data_cog_gray_nr[2],
                        "CGG_NR_ANGLE": data_cog_gray_nr[3],
                    }
                )
            # Obtenemos centro de gravedad de los bordes de la imagen
            if EXTRACT_COG_CANNY:
                data_cog_canny = cls.center_of_gravity_info(edges)
                data_cog_canny_nr = cls.center_of_gravity_info(edges_without_ring)
                print(
                    f"Edges COG Info -> Dist ({round(data_cog_canny[2],2)}) Angle ({round(data_cog_canny[3],2)})"
                )
                data.update(
                    {
                        "CGC_X": data_cog_canny[0],
                        "CGC_Y": data_cog_canny[1],
                        "CGC_DIST": data_cog_canny[2],
                        "CGC_ANGLE": data_cog_canny[3],
                        "CGC_NR_X": data_cog_canny_nr[0],
                        "CGC_NR_Y": data_cog_canny_nr[1],
                        "CGC_NR_DIST": data_cog_canny_nr[2],
                        "CGC_NR_ANGLE": data_cog_canny_nr[3],
                    }
                )

            # Angulo entre el cog de los bordes y el cog de los keypoints
            if EXTRACT_COG_CANNY and EXTRACT_KEYPWR:
                centers_coords = (
                    data_cog_canny[0],
                    data_cog_canny[1],
                    data_keyP[1],
                    data_keyP[2],
                )
                centers_coords_nr = (
                    data_cog_canny_nr[0],
                    data_cog_canny_nr[1],
                    data_keyP_nr[1],
                    data_keyP_nr[2],
                )
                data_cgc_ckp = {
                    "CGC_CKP_ANGLE1": abs(data_keyP[4] - data_cog_canny[3]),
                    "CGC_CKP_ANGLE2": dp.angle_func(centers_coords),
                    "CGC_CKP_LONG": dp.len_func(centers_coords),
                    "NR_CGC_CKP_ANGLE1": abs(data_keyP_nr[4] - data_cog_canny_nr[3]),
                    "NR_CGC_CKP_ANGLE2": dp.angle_func(centers_coords_nr),
                    "NR_CGC_CKP_LONG": dp.len_func(centers_coords_nr),
                }
                data.update(data_cgc_ckp)

                print(f"CGC and CKP Info -> {data_cgc_ckp}")

            # Obtenemos la similitud con diferentes anillos
            if EXTRACT_RINGSIMS:
                data_ring_similarities = cls.compareImgs(ring, rings2compare)
                # print(data_ring_similarities)
                data.update(data_ring_similarities)

            if ESTRACT_PIXELS_DATA:
                data.update(
                    {
                        "EDGES_COUNT": np.sum(edges == 255),
                        "LIGHT_PIXELS_COUNT": np.sum(gray > 150),
                        "EDGES_COUNT_NORING": np.sum(edges_without_ring == 255),
                        "LIGHT_PIXELS_COUNT_NORING": np.sum(gray_without_ring > 150),
                    }
                )

            data["CLASS"] = class_id  # Tambien guardamos la id de la clase

            for key in all_data:
                if data.get(key) is not None:
                    all_data[key].append(data[key])
                else:
                    all_data[key].append(None)

            if DEBUG:
                plt.subplot(241)
                plt.title("Original")
                plt.imshow(img)

                plt.subplot(242)
                plt.title("Cropped")
                plt.imshow(image)

                if ncoins > 0:
                    plt.subplot(243)
                    plt.title("Circles")
                    plt.imshow(cls.DCircles)

                plt.show()

        return all_data

    @classmethod
    def cropCircle(cls, img: cv2.Mat, ncoins=1) -> list[cv2.Mat]:

        # Si el fondo es blanco hacemos correcion de sombras
        _, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        height, width = img.shape[:2]
        # Aplicamos un suavizado para eliminar ruidos
        blurred = cls.gaussBlur(img)

        _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        threshold = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        morphed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
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

        # Obtenemos la mediana de los radios

        rcoin = circles[0, 0, 2]
        print(f"Coin radius (px) -> {rcoin}")

        cropped_ls = []
        # Pasamos a escala BGR para pocer marcar la imagen con colores
        if DEBUG:
            show = cv2.cvtColor(morphed.copy(), cv2.COLOR_GRAY2BGR)
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
                print(f"Bad circle: x={x} y={y} r={r}")
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
        # Muestra cuantos circulos se han encontrado
        print(f"Extracted {ncoins} circles")

        return cropped_ls

    @staticmethod
    def getOCR(img: cv2.Mat) -> list[str]:
        h, w = img.shape[:2]
        # Creamos el reader
        reader = easyocr.Reader(["en"])
        # Leemos la moneda
        output = reader.readtext(img)
        M = cv2.getRotationMatrix2D((h / 2, w / 2), 360 / dp.OCR_N_READS, 1)
        for _ in range(dp.OCR_N_READS - 1):
            rotated = cv2.warpAffine(img, M, (w, h))
            output += reader.readtext(rotated)

        palabras = set()
        copy = img.copy()
        print(f"Good words ->", end=" ")
        for tupla in output:
            cord, palabra, percent = tupla
            if percent > dp.OCR_MINRATE:
                # -- DEBUG --
                #  Muestra  donde detecto los caracteres
                x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
                x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
                cv2.rectangle(copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                # -------------

                # Si el indice de acierto es > 70%
                palabras.add(palabra)
                print(f"{palabra} ({round(percent,2)})", end=" ")
        print()
        if DEBUG:
            plt.subplot(244)
            plt.imshow(copy)
            plt.title("OCR")

        return palabras

    @staticmethod
    def huMoments(img: cv2.Mat) -> list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculamos los momentos estadisticos hasta los de primera orden
        M = cv2.moments(gray, False)
        # Calculamos los momentos de Hu y nos quedamos con los dos primeros
        Hm = cv2.HuMoments(M).flatten()[0 : dp.N_HUMOMS]
        return Hm

    @classmethod
    def keyPoints(cls, img: cv2.Mat):
        # Detección de esquinas
        kps_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps_img = cls.gaussBlur(kps_img)

        kps = cv2.goodFeaturesToTrack(
            np.float32(kps_img), dp.KP_MAXCORNERS, dp.KP_QUALITY, dp.KP_MINDIST
        )
        kps_img = cv2.cvtColor(kps_img, cv2.COLOR_GRAY2BGR)

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
        return len(kps), x, y, dist, angle

    @classmethod
    def getLines(cls, edges: cv2.Mat):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morphed = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)

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
    def normalize_orientation(cls, dist, angle, img: cv2.Mat):
        h, w = img.shape
        cy, cx = h / 2, w / 2
        if dist > dp.MIN_CENTERS_DIST:
            M = cv2.getRotationMatrix2D((cx, cy), angle + 180, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
        else:
            rotated = img

        if DEBUG:
            plt.subplot(247)
            plt.imshow(rotated, "gray")
            plt.title("Rotated")

        return rotated

    @classmethod
    def center_of_gravity_info(cls, img: cv2.Mat):
        h, w = img.shape
        cy, cx = h / 2, w / 2

        cmy, cmx = ndi.center_of_mass(img)

        dist = np.sqrt((cx - cmx) ** 2 + (cy - cmy) ** 2) * dp.IM_SIZE_ADJ / h
        angle = math.degrees(math.atan2(cy - cmy, cx - cmx)) % 360

        if DEBUG:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            plt.subplot(246)
            cv2.circle(img, (int(cx), int(cy)), 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (int(cmx), int(cmy)), 5, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (int(cx), int(cy)), (int(cmx), int(cmy)), (0, 0, 255), 2)
            plt.imshow(img)
            plt.title("Centers")

        return cmx, cmy, dist, angle

    @classmethod
    def compareImgs(cls, img1: cv2.Mat, img2_ls: list[cv2.Mat]):
        img1 = cv2.resize(img1, (512, 512))

        data = {}
        # print(f"Ring similarities ->", end=" ")
        for i, img2 in enumerate(img2_ls, start=1):

            data[f"RING_MSE_{i}"] = mse(img1, img2)
            data[f"RING_RMSE_{i}"] = rmse(img1, img2)
            data[f"RING_PSNR_{i}"] = psnr(img1, img2)
            data[f"RING_UQI_{i}"] = uqi(img1, img2)
            data[f"RING_SSIM_{i}"] = ssim(img1, img2)[1]
            data[f"RING_ERGAS_{i}"] = ergas(img1, img2)
            data[f"RING_SCC_{i}"] = scc(img1, img2)
            # data[f"RING_RASE_{i}"] = rase(img1, img2)
            data[f"RING_SAM_{i}"] = sam(img1, img2)
            data[f"RING_MSSSIM_{i}"] = np.real(msssim(img1, img2))
            data[f"RING_VIF_{i}"] = vifp(img1, img2)
            # cls.quickShow(np.concatenate((img1, img2), axis=1))
        return data
