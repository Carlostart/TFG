import math
from pathlib import Path
from difflib import SequenceMatcher as SM
import numpy as np

PROJECT_ROOT = Path(__file__).parents[1]

FILE_PATHS = PROJECT_ROOT / 'data' / "path_to_data.txt"
FILE_CSV = PROJECT_ROOT / 'data' / "DataSet.csv"
FILE_COIN_INFO = PROJECT_ROOT / 'data' / "coin_info.txt"
FILE_COIN_WORDS = PROJECT_ROOT / 'data' / "coin_words.txt"
OUT_FOLDER = 'C:/Users/Carlo/OneDrive/Escritorio/out'

IM_SIZE_ADJ = 255

CANNY_TRHES1 = 50
CANNY_TRHES2 = 100

HCIRCLES_KERNEL_RATIO = 100
HCIRCLES_DP = 4
HCIRCLES_PAR1 = 50
HCIRCLES_PAR2 = 100
HCIRCLES_MINRAD = 127

HLINES_KERNEL_RATIO = 60

OCR_MINRATE = 0.75

N_HUMOMS = 2

KP_MAXCORNERS = 10
KP_QUALITY = 0.1
KP_MINDIST = 5

NUM_LINES = 20

MIN_CENTERS_DIST = 10

with open(FILE_COIN_WORDS, 'r') as f:
    COIN_WORDS = set(f.readlines())


def initData():
    data = {"HU_1": [],
            "HU_2": [],
            "CG_X": [],
            "CG_Y": [],
            "BEST_AVG_LEN": [],
            "BEST_AVG_ANGLE": [],
            "BEST_AVG_X": [],
            "BEST_AVG_Y": [],
            "BEST_VAR_LEN": [],
            "BEST_VAR_ANGLE": [],
            "LONG_AVG_LEN": [],
            "LONG_AVG_ANGLE": [],
            "LONG_AVG_X": [],
            "LONG_AVG_Y": [],
            "LONG_VAR_ANGLE": [],
            "LONG_VAR_LEN": [],
            "LONGEST_LEN": [],
            "LONGEST_ANGLE": [],
            "LONGEST_X": [],
            "LONGEST_Y": []
            }
    for word in COIN_WORDS:
        data.update({f"OCR_{word[:-1]}": []})
    data.update({"CLASS": []})
    return data


def appendOcrData(ocr_data, data):
    for word in COIN_WORDS:
        m = 0
        for ocr in ocr_data:
            aux = SM(None, word[:-1], ocr).ratio()
            m = max(m, aux)
        data.get(f"OCR_{word[:-1]}").append(m)


def getLinesData(lines: list, img_size: int):
    lines_data = {}
    if lines is not None:

        def len_func(x): return np.sqrt(
            (x[0]-x[2])**2 + (x[1]-x[3])**2)

        def angle_func(x):
            angle = math.degrees(math.atan2(x[1]-x[3], x[0]-x[2])) % 360
            return angle-180 if angle > 180 else angle

        lines = lines*IM_SIZE_ADJ/img_size
        sorted_lines = np.array(
            sorted(lines, key=len_func, reverse=True))

        best_lines = lines[:NUM_LINES]
        long_lines = sorted_lines[:NUM_LINES]

        best_lens = [len_func(line) for line in best_lines]
        best_angles = [angle_func(line) for line in best_lines]
        long_lens = [len_func(line) for line in long_lines]
        long_angles = [angle_func(line) for line in long_lines]

        lines_data["BEST_AVG_LEN"] = np.mean(best_lens)
        lines_data["BEST_AVG_ANGLE"] = np.mean(best_angles)
        lines_data["BEST_AVG_X"] = (
            np.mean(best_lines[:, 0]) + np.mean(best_lines[:, 2]))/2
        lines_data["BEST_AVG_Y"] = (
            np.mean(best_lines[:, 1]) + np.mean(best_lines[:, 3]))/2
        lines_data["BEST_VAR_LEN"] = np.var(best_lens)
        lines_data["BEST_VAR_ANGLE"] = np.var(best_angles)

        lines_data["LONG_AVG_LEN"] = np.mean(long_lens)
        lines_data["LONG_AVG_ANGLE"] = np.mean(long_angles)
        lines_data["LONG_AVG_X"] = (
            np.mean(long_lines[:, 0]) + np.mean(long_lines[:, 2]))/2
        lines_data["LONG_AVG_Y"] = (
            np.mean(long_lines[:, 1]) + np.mean(long_lines[:, 3]))/2
        lines_data["LONG_VAR_LEN"] = np.var(long_lens)
        lines_data["LONG_VAR_ANGLE"] = np.var(long_angles)

        longest = long_lines[0]
        lines_data["LONGEST_LEN"] = len_func(longest)
        lines_data["LONGEST_ANGLE"] = angle_func(longest)
        lines_data["LONGEST_X"] = (longest[0]+longest[2])/2
        lines_data["LONGEST_Y"] = (longest[1]+longest[3])/2

    return lines_data


def getClass(file_name):
    # Si hay multiples monedas en la imagen debe estar indicado entre corchetes
    class_id = file_name.split(' ')[0]
    ncoins = 1
    if file_name[0] == '[':
        aux = class_id.split(']')
        ncoins = int(aux[0][1:])
        class_id = aux[1]
    return class_id, ncoins
