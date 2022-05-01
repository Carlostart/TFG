from difflib import SequenceMatcher as SM

FILE_PATHS = "path_to_data.txt"
FILE_CSV = "DataSet.csv"
FILE_COIN_INFO = "coin_info.txt"

IM_SIZE_ADJ = 255


CANNY_TRHES1 = 50
CANNY_TRHES2 = 100

HCIRCLES_GAUSS_KERNEL = (19, 19)
HCIRCLES_GAUSS_SIGMA = 5
HCIRCLES_DP = 4
HCIRCLES_PAR1 = 50
HCIRCLES_PAR2 = 100
HCIRCLES_MINRAD = 127

HLINES_GAUSS_KERNEL = (7, 7)
HLINES_GAUSS_SIGMA = 5

OCR_MINRATE = 0.7

N_HUMOMS = 2

KP_MAXCORNERS = 10
KP_QUALITY = 0.1
KP_MINDIST = 5

with open("coin_words.txt", 'r') as f:
    COIN_WORDS = set(f.readlines())


def initData():
    data = {"HU_1": [],
            "HU_2": []
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
