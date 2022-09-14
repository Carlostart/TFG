import data_processing as dp
import im_processing as ip
import matplotlib.pyplot as plt
import numpy as np
import cv2
from config_params import *

# EXTRACT_OCR = True
# EXTRACT_RINGSIMS = True

# NORMALIZE_ORIENTATION = False

# EXTRACT_HU = True
# EXTRACT_KEYPWR = True
# EXTRACT_COG_GRAY = True
# EXTRACT_COG_CANNY = True
# EXTRACT_LINES = True
# ESTRACT_PIXELS_DATA = True
# DEBUG = False


def extractData(img_path: str, ncoins=1) -> dict[str, list]:
    print(
        "==========================================================\n"
        + f"Identificando imagen: {img_path}"
    )
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    # img = cv2.imread(img_path)[:, :, ::-1]  # Lee imagen
    # -- Ajustes de imagen para que sigan un mismo formato --
    if ncoins > 0:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # Separa y recorta cada moneda
        images = ip.cropCircles(img, ncoins, DEBUG)
    else:
        images = [img]

    # Si hay varias monedas en una misma imagen procesamos todas
    rings2compare = [cv2.imread(ring) for ring in dp.getFilesInFolders([RINGS_FOLDER])]
    for image in images:
        data = {}
        # Aplicamos CLAHE y eliminamos el borde exterior de la moneda
        image = ip.clahe(image)
        reduced = ip.reduceCircle(image, 0.9)
        ring = ip.getOuterRing(reduced, RING_SIZE)
        img_without_ring = ip.reduceCircle(reduced, RING_SIZE)

        # Procesamos la imagen ajustando el tamaño, haciendo escala de grises y calculando bordes
        resized = cv2.resize(reduced, (IM_SIZE_ADJ, IM_SIZE_ADJ))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_ring = cv2.cvtColor(ring, cv2.COLOR_BGR2GRAY)
        gray_without_ring = ip.reduceCircle(gray, RING_SIZE)

        edges = ip.edgesInside(reduced)
        sobel_edges = ip.edgesSobel(reduced)
        sobel_edges_nr = ip.reduceCircle(gray, RING_SIZE)
        sobel_edges_ring = ip.getOuterRing(reduced, RING_SIZE)

        # Separamos el anillo de la moneda en diferentes formatos
        ring_edges = ip.getOuterRing(edges, RING_SIZE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
        edges_without_ring = ip.reduceCircle(edges, RING_SIZE)
        # Empezamos con la extracción de datos
        # Momentos de Hu
        if EXTRACT_HU:
            data_Hu = ip.huMoments(resized)
            print(f"Hu Moments ->", end=" ")
            for i in range(N_HUMOMS):
                print(data_Hu[i], end=" ")
                data.update({f"HU_{i+1}": data_Hu[i]})
            print()
        # Lectura de caracteres
        if EXTRACT_OCR:
            data_ocr = ip.getOCR(image, DEBUG)
            dp.appendOcrData(data_ocr, data)
        # Obtenemos info delas esquinas
        if EXTRACT_KEYPWR:
            data_keyP_r = ip.keyPoints(ring, DEBUG)  # WITH RING
            data_keyP_nr = ip.keyPoints(img_without_ring)  # WITH RING
            print(
                f"KeyPs Center Info -> Dist ({round(data_keyP_nr[3],2)}) Angle ({round(data_keyP_r[4],2)})"
            )
            data.update(
                {
                    "CKP_R_N": data_keyP_r[0],
                    "CKP_R_X": data_keyP_r[1],
                    "CKP_R_Y": data_keyP_r[2],
                    "CKP_R_DIST": data_keyP_r[3],
                    "CKP_R_ANGLE": data_keyP_r[4],
                    "CKP_NR_N": data_keyP_nr[0],
                    "CKP_NR_X": data_keyP_nr[1],
                    "CKP_NR_Y": data_keyP_nr[2],
                    "CKP_NR_DIST": data_keyP_nr[3],
                    "CKP_NR_ANGLE": data_keyP_nr[4],
                }
            )

            # Normalizamos la orientación de las imagenes que usaremos
            if NORMALIZE_ORIENTATION:
                edges_without_ring = ip.normalize_orientation(
                    data_keyP_r[2], data_keyP_r[3], edges_without_ring
                )
                gray_without_ring = ip.normalize_orientation(
                    data_keyP_r[2], data_keyP_r[3], gray_without_ring
                )
                gray = ip.normalize_orientation(data_keyP_r[2], data_keyP_r[3], gray)
                edges = ip.normalize_orientation(data_keyP_r[2], data_keyP_r[3], edges)

        # Obtenemos información de las líneas
        if EXTRACT_LINES:
            lines = ip.getLines(edges_without_ring, DEBUG)
            data_lines = dp.getLinesData(lines, edges_without_ring.shape[0])
            # print(f"Lines Info -> {data_lines}")
            data.update(data_lines)

        # Obtenemos centro de gravedad de la imagen en escala de grises
        if EXTRACT_COG_GRAY:
            data_cog_gray_r = ip.center_of_gravity_info(gray_ring, DEBUG)
            data_cog_gray_nr = ip.center_of_gravity_info(gray_without_ring)
            print(
                f"Gray COG Info -> Dist ({round(data_cog_gray_r[2],2)}) Angle ({round(data_cog_gray_r[3],2)})"
            )
            data.update(
                {
                    "CGG_R_X": data_cog_gray_r[0],
                    "CGG_R_Y": data_cog_gray_r[1],
                    "CGG_R_DIST": data_cog_gray_r[2],
                    "CGG_R_ANGLE": data_cog_gray_r[3],
                    "CGG_NR_X": data_cog_gray_nr[0],
                    "CGG_NR_Y": data_cog_gray_nr[1],
                    "CGG_NR_DIST": data_cog_gray_nr[2],
                    "CGG_NR_ANGLE": data_cog_gray_nr[3],
                }
            )
        # Obtenemos centro de gravedad de los bordes de la imagen
        if EXTRACT_COG_CANNY:
            # data_cog_canny = ip.center_of_gravity_info(edges)
            data_cog_canny_nr = ip.center_of_gravity_info(edges_without_ring)
            data_cog_canny_r = ip.center_of_gravity_info(ring_edges)
            print(
                f"Edges COG Info -> Dist ({round(data_cog_canny_r[2],2)}) Angle ({round(data_cog_canny_r[3],2)})"
            )
            data.update(
                {
                    # "CGC_X": data_cog_canny[0],
                    # "CGC_Y": data_cog_canny[1],
                    # "CGC_DIST": data_cog_canny[2],
                    # "CGC_ANGLE": data_cog_canny[3],
                    "CGC_NR_X": data_cog_canny_nr[0],
                    "CGC_NR_Y": data_cog_canny_nr[1],
                    "CGC_NR_DIST": data_cog_canny_nr[2],
                    "CGC_NR_ANGLE": data_cog_canny_nr[3],
                    "CGC_R_X": data_cog_canny_r[0],
                    "CGC_R_Y": data_cog_canny_r[1],
                    "CGC_R_DIST": data_cog_canny_r[2],
                    "CGC_R_ANGLE": data_cog_canny_r[3],
                }
            )

        # Angulo entre el cog de los bordes y el cog de los keypoints
        if EXTRACT_COG_CANNY and EXTRACT_KEYPWR:
            centers_coords = (
                data_cog_canny_r[0],
                data_cog_canny_r[1],
                data_keyP_r[1],
                data_keyP_r[2],
            )
            centers_coords_nr = (
                data_cog_canny_nr[0],
                data_cog_canny_nr[1],
                data_keyP_nr[1],
                data_keyP_nr[2],
            )
            data_cgc_ckp = {
                "CGC_CKP_ANGLE1": abs(data_keyP_r[4] - data_cog_canny_r[3]),
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
            data_ring_similarities = ip.compareImgs(ring, rings2compare)
            # print(data_ring_similarities)
            data.update(data_ring_similarities)

        if ESTRACT_PIXELS_DATA:
            data.update(
                {
                    "CANNY_PX_COUNT_R": np.sum(ring_edges == 255),
                    "SOBEL_PX_COUNT_R": np.sum(sobel_edges_ring) / 255,
                    "CANNY_PX_COUNT_NR": np.sum(edges_without_ring == 255),
                    "SOBEL_PX_COUNT_NR": np.sum(sobel_edges_nr) / 255,
                }
            )

        if DEBUG:
            plt.subplot(241)
            plt.title("Original")
            plt.imshow(img)

            plt.subplot(242)
            plt.title("Cropped")
            plt.imshow(image)

            plt.subplot(243)
            plt.title("Edges")
            plt.imshow(img_without_ring, "gray")

            plt.subplot(247)
            plt.title("Edges")
            plt.imshow(edges, "gray")

            plt.show()

    return data
