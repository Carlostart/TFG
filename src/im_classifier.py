import data_processing as dp
import im_processing as ip
import matplotlib.pyplot as plt
import numpy as np
import cv2

EXTRACT_OCR = False
EXTRACT_RINGSIMS = False

NORMALIZE_ORIENTATION = True

EXTRACT_HU = True
EXTRACT_KEYPWR = True
EXTRACT_COG_GRAY = True
EXTRACT_COG_CANNY = True
EXTRACT_LINES = True
ESTRACT_PIXELS_DATA = True
DEBUG = False


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
        images = ip.cropCircle(img, ncoins, DEBUG)
    else:
        images = [img]

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
        equalized = ip.clahe(image)
        reduced = ip.removeExternalRing(image, 0.9)
        img_without_ring = ip.removeExternalRing(image, 0.72)
        # Procesamos la imagen ajustando el tamaño, haciendo escala de grises y calculando bordes
        resized = cv2.resize(reduced, (dp.IM_SIZE_ADJ, dp.IM_SIZE_ADJ))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_without_ring = ip.removeExternalRing(gray, 0.72)

        edges = ip.edgesInside(reduced)
        # Separamos el anillo de la moneda en diferentes formatos
        ring = ip.getOuterRing(reduced, 0.72)
        ring_edges = ip.getOuterRing(edges, 0.72)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
        edges_without_ring = ip.removeExternalRing(edges, 0.72)
        # Empezamos con la extracción de datos
        # Momentos de Hu
        if EXTRACT_HU:
            data_Hu = ip.huMoments(resized)
            print(f"Hu Moments ->", end=" ")
            for i in range(dp.N_HUMOMS):
                print(data_Hu[i], end=" ")
                data.update({f"HU_{i+1}": data_Hu[i]})
            print()
        # Lectura de caracteres
        if EXTRACT_OCR:
            data_ocr = ip.getOCR(equalized, DEBUG)
            dp.appendOcrData(data_ocr, data)
        # Obtenemos info delas esquinas
        if EXTRACT_KEYPWR:
            data_keyP = ip.keyPoints(reduced, DEBUG)  # WITH RING
            data_keyP_nr = ip.keyPoints(img_without_ring)  # WITH RING
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
                edges_without_ring = ip.normalize_orientation(
                    data_keyP[2], data_keyP[3], edges_without_ring
                )
                gray_without_ring = ip.normalize_orientation(
                    data_keyP[2], data_keyP[3], gray_without_ring
                )
                gray = ip.normalize_orientation(data_keyP[2], data_keyP[3], gray)
                edges = ip.normalize_orientation(data_keyP[2], data_keyP[3], edges)

        # Obtenemos información de las líneas
        if EXTRACT_LINES:
            lines = ip.getLines(edges_without_ring, DEBUG)
            data_lines = dp.getLinesData(lines, edges_without_ring.shape[0])
            # print(f"Lines Info -> {data_lines}")
            data.update(data_lines)

        # Obtenemos centro de gravedad de la imagen en escala de grises
        if EXTRACT_COG_GRAY:
            data_cog_gray = ip.center_of_gravity_info(gray, DEBUG)
            data_cog_gray_nr = ip.center_of_gravity_info(gray_without_ring)
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
            data_cog_canny = ip.center_of_gravity_info(edges)
            data_cog_canny_nr = ip.center_of_gravity_info(edges_without_ring)
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
            data_ring_similarities = ip.compareImgs(ring, rings2compare)
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

        if DEBUG:
            plt.subplot(241)
            plt.title("Original")
            plt.imshow(img)

            plt.subplot(242)
            plt.title("Cropped")
            plt.imshow(image)

            plt.show()

    return data
