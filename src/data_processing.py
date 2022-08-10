import math
import os
from pathlib import Path
from difflib import SequenceMatcher as SM
import numpy as np

PROJECT_ROOT = Path(__file__).parents[1]

FILE_PATHS = PROJECT_ROOT / "conf" / "path_to_data.txt"
FILE_CSV = PROJECT_ROOT / "data" / "DataSet.csv"
FILE_COIN_INFO = PROJECT_ROOT / "conf" / "coin_info.txt"
FILE_COIN_WORDS = PROJECT_ROOT / "conf" / "coin_words.txt"
OUT_FOLDER = PROJECT_ROOT / "data"
RINGS_FOLDER = "conf/RINGS"

IM_SIZE_ADJ = 512

CANNY_TRHES1 = 50
CANNY_TRHES2 = 100

HCIRCLES_KERNEL_RATIO = 100
HCIRCLES_DP = 4
HCIRCLES_PAR1 = 50
HCIRCLES_PAR2 = 100
HCIRCLES_MINRAD = 127

HLINES_KERNEL_RATIO = 35

OCR_MINRATE = 0.75
OCR_N_READS = 4

N_HUMOMS = 7

KP_MAXCORNERS = 1000
KP_QUALITY = 0.1
KP_MINDIST = 5

NUM_LINES = 20

MIN_CENTERS_DIST = 500

SIFT_PERCENTAGE_FOR_GP = 0.7

# with open(FILE_COIN_WORDS, "r") as f:
#     COIN_WORDS = set(f.readlines())

OCR_CHARS = [
    chr(i) for i in [*range(ord("A"), ord("Z") + 1), *range(ord("0"), ord("9") + 1)]
]


def getFilesInFolders(pths):
    for pth in pths:
        if os.path.isdir(pth):
            pths.remove(pth)
            # Introducimos todos los ficheros de la carpeta junto con su path a la lists
            pths += [os.path.join(pth, file) for file in os.listdir(pth)]

    return pths


RING_FILES = getFilesInFolders([RINGS_FOLDER])


def initData():
    data = {
        "HU_1": [],
        "HU_2": [],
        "CGG_X": [],
        "CGG_Y": [],
        "CGG_DIST": [],
        "CGG_ANGLE": [],
        "CGC_X": [],
        "CGC_Y": [],
        "CGC_DIST": [],
        "CGC_ANGLE": [],
        "CKP_N": [],
        "CKP_X": [],
        "CKP_Y": [],
        "CKP_DIST": [],
        "CKP_ANGLE": [],
        "CGC_CKP_ANGLE1": [],
        "CGC_CKP_ANGLE2": [],
        "CGC_CKP_LONG": [],
        "BEST_AVG_X": [],
        "BEST_AVG_Y": [],
        "BEST_AVG_LEN": [],
        "BEST_AVG_ANGLE": [],
        "BEST_VAR_LEN": [],
        "BEST_VAR_ANGLE": [],
        "LONG_AVG_X": [],
        "LONG_AVG_Y": [],
        "LONG_AVG_LEN": [],
        "LONG_AVG_ANGLE": [],
        "LONG_VAR_LEN": [],
        "LONG_VAR_ANGLE": [],
        "LONGEST_X": [],
        "LONGEST_Y": [],
        "LONGEST_LEN": [],
        "LONGEST_ANGLE": [],
    }
    # for word in COIN_WORDS:
    #     data.update({f"OCR_{word[:-1]}": []})
    for letter in OCR_CHARS:
        data.update({f"OCR_{letter}": []})
    for number in range(10):
        data.update({f"OCR_{number}": []})

    for i in range(1, len(RING_FILES) + 1):
        data.update({f"RING_MSE_{i}": []})
        data.update({f"RING_RMSE_{i}": []})
        data.update({f"RING_PSNR_{i}": []})
        data.update({f"RING_UQI_{i}": []})
        data.update({f"RING_SSIM_{i}": []})
        data.update({f"RING_ERGAS_{i}": []})
        data.update({f"RING_SCC_{i}": []})
        # data.update({f"RING_RASE_{i}": []})
        data.update({f"RING_SAM_{i}": []})
        data.update({f"RING_MSSSIM_{i}": []})
        data.update({f"RING_VIF_{i}": []})
    data.update({"CLASS": []})
    return data


def getClass(pth):
    file = pth.split("\\")[-1]
    file = file.split("/")[-1]
    class_id = file.split(" ")[0]
    if "[" in file:
        aux = class_id.split("]")
        ncoins = int(aux[0][1:])
        class_id = aux[1]
    else:
        ncoins = 1
    if "CROPPED" in pth:
        ncoins = -1
    return class_id, ncoins


# def appendOcrData(ocr_data, data):
#     for word in COIN_WORDS:
#         m = 0
#         for ocr in ocr_data:
#             aux = SM(None, word[:-1], ocr).ratio()
#             m = max(m, aux)
#         data.get(f"OCR_{word[:-1]}").append(m)


def appendOcrData(ocr_data, data):
    for ch in OCR_CHARS:
        count = 0
        for ocr in ocr_data:
            if ch in ocr:
                count += 1
        data[f"OCR_{ch}"] = count


def len_func(x):
    return np.sqrt((x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2)


def angle_func(x):
    angle = math.degrees(math.atan2(x[1] - x[3], x[0] - x[2])) % 360
    return angle - 180 if angle > 180 else angle


def getLinesData(lines: list, img_size: int):
    lines_data = {}
    if lines is not None:

        lines = lines * IM_SIZE_ADJ / img_size
        sorted_lines = np.array(sorted(lines, key=len_func, reverse=True))

        best_lines = lines[:NUM_LINES]
        long_lines = sorted_lines[:NUM_LINES]

        best_lens = [len_func(line) for line in best_lines]
        best_angles = [angle_func(line) for line in best_lines]
        long_lens = [len_func(line) for line in long_lines]
        long_angles = [angle_func(line) for line in long_lines]

        lines_data["BEST_AVG_LEN"] = np.mean(best_lens)
        lines_data["BEST_AVG_ANGLE"] = np.mean(best_angles)
        lines_data["BEST_AVG_X"] = (
            np.mean(best_lines[:, 0]) + np.mean(best_lines[:, 2])
        ) / 2
        lines_data["BEST_AVG_Y"] = (
            np.mean(best_lines[:, 1]) + np.mean(best_lines[:, 3])
        ) / 2
        lines_data["BEST_VAR_LEN"] = np.var(best_lens)
        lines_data["BEST_VAR_ANGLE"] = np.var(best_angles)

        lines_data["LONG_AVG_LEN"] = np.mean(long_lens)
        lines_data["LONG_AVG_ANGLE"] = np.mean(long_angles)
        lines_data["LONG_AVG_X"] = (
            np.mean(long_lines[:, 0]) + np.mean(long_lines[:, 2])
        ) / 2
        lines_data["LONG_AVG_Y"] = (
            np.mean(long_lines[:, 1]) + np.mean(long_lines[:, 3])
        ) / 2
        lines_data["LONG_VAR_LEN"] = np.var(long_lens)
        lines_data["LONG_VAR_ANGLE"] = np.var(long_angles)

        longest = long_lines[0]
        lines_data["LONGEST_LEN"] = len_func(longest)
        lines_data["LONGEST_ANGLE"] = angle_func(longest)
        lines_data["LONGEST_X"] = (longest[0] + longest[2]) / 2
        lines_data["LONGEST_Y"] = (longest[1] + longest[3]) / 2

    return lines_data
