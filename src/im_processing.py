import data_processing as dp

import scipy.ndimage as ndi
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr
import PIL
from PIL import Image, ImageTk

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


def quickShow(img, title=""):
    plt.imshow(img, "gray")
    plt.title(title)
    plt.show()


def reduceCircle(img: cv2.Mat, pr) -> cv2.Mat:
    height, width = img.shape[:2]
    # Ahora eliminamos el anillo exterior de la moneda
    mask = np.zeros((height, width), np.uint8)
    x, y, r = int(height / 2), int(width / 2), int((height / 2) * pr)

    cv2.circle(
        mask,
        (x, y),
        r,
        255,
        thickness=-1,
    )
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked[y - r : y + r, x - r : x + r]


def getOuterRing(img: cv2.Mat, pr) -> cv2.Mat:
    height, width = img.shape[:2]
    mask = np.ones((height, width), np.uint8)
    x, y, r = int(height / 2), int(width / 2), int((height / 2) * pr)
    cv2.circle(
        mask,
        (x, y),
        r,
        0,
        thickness=-1,
    )
    return cv2.bitwise_and(img, img, mask=mask)


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


def edgesInside(img):
    aux = img
    for _ in range(2):
        # Aplicamos un suavizado para eliminar ruidos
        aux = gaussBlur(aux)
        aux = clahe(aux)
    # Obtenemos los bordes de la imagen
    edges = cv2.Canny(aux, dp.CANNY_TRHES1, dp.CANNY_TRHES2)

    return edges


def edgesSobel(gray_img):

    ksize = -1
    gX = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(gray_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    # on them and visualize them
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    # combine the gradient representations into a single image
    return cv2.addWeighted(gX, 0.5, gY, 0.5, 0)


# https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df
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

    output = np.row_stack([output_B, output_G, output_R]).T.reshape(height, width, 3)
    output = output = np.minimum(np.maximum(output, 0), 255).astype(np.uint8)
    return output


def cropCircles(img: cv2.Mat, ncoins=1, DEBUG=False) -> list[cv2.Mat]:

    height, width = img.shape[:2]
    # Aplicamos un suavizado para eliminar ruidos
    blurred = gaussBlur(img)
    # Pasamos a escala de grises
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Aplicamos detección de bordes con Sobel
    edges_sobel = edgesSobel(gray)
    # Aplicamos morfologia de cierre para definir mejor la siueta de la moneda
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    morphed = cv2.morphologyEx(edges_sobel, cv2.MORPH_CLOSE, kernel, iterations=1)

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
    best_radious = circles[0, 0, 2]
    print(f"Coin radius (px) -> {best_radious}")

    cropped_ls = []

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
        if r > best_radious * 1.2 or r < best_radious * 0.8:
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
            # Pasamos a escala BGR para pocer marcar la imagen con colores
            show = cv2.cvtColor(morphed.copy(), cv2.COLOR_GRAY2BGR)
            cv2.circle(show, (x, y), r, (255, 0, 0), thickness=10)
            plt.subplot(243)
            plt.title("Circles")
            plt.imshow(show)
        #  ------------
        # Nos salimos del bucle cuando hemos seleccionado el numero de monedas especificado
        if idx >= ncoins - 1 and (deleted >= 3 or not miss):
            break
    # Muestra cuantos circulos se han encontrado
    print(f"Extracted {ncoins} circles")

    return cropped_ls


def getOCR(img: cv2.Mat, DEBUG=False, n_reads=1) -> list[str]:
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


def huMoments(img: cv2.Mat) -> list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculamos los momentos estadisticos hasta los de primera orden
    M = cv2.moments(gray, False)
    # Calculamos los momentos de Hu y nos quedamos con los dos primeros
    Hm = cv2.HuMoments(M).flatten()[0 : dp.N_HUMOMS]
    return Hm


def keyPoints(img: cv2.Mat, DEBUG=False):
    # Detección de esquinas
    kps_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps_img = gaussBlur(kps_img)

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


def getLines(edges: cv2.Mat, DEBUG=False):
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


def normalize_orientation(dist, angle, img: cv2.Mat):
    h, w = img.shape
    cy, cx = h / 2, w / 2
    if dist > dp.MIN_CENTERS_DIST:
        M = cv2.getRotationMatrix2D((cx, cy), angle + 180, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
    else:
        rotated = img

    return rotated


def center_of_gravity_info(img: cv2.Mat, DEBUG=False):
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


def compareImgs(img1: cv2.Mat, img2_ls: list[cv2.Mat]):
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
        # quickShow(np.concatenate((img1, img2), axis=1))
    return data


def cv2_to_imageTK(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    imagePIL = PIL.Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=imagePIL)
    return imgtk
