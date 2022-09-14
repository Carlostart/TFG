# ===========================================
# ------------- EXTRACTORS ------------------
# ===========================================
EXTRACT_OCR = True
EXTRACT_RINGSIMS = True

NORMALIZE_ORIENTATION = False

EXTRACT_HU = True
EXTRACT_KEYPWR = True
EXTRACT_COG_GRAY = True
EXTRACT_COG_CANNY = True
EXTRACT_LINES = True
ESTRACT_PIXELS_DATA = True
DEBUG = False


# ===========================================
# ----------------- PATHS -------------------
# ===========================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]

FILE_PATHS = PROJECT_ROOT / "conf" / "path_to_data.conf"
FILE_ARFF = PROJECT_ROOT / "data" / "DataSet.arff"
FILE_MODEL = PROJECT_ROOT / "data" / "models" / "SimpleLogistic.model"
FILE_COIN_INFO = PROJECT_ROOT / "conf" / "coin_info.conf"
FILE_COIN_WORDS = PROJECT_ROOT / "conf" / "coin_words.conf"
OUT_FOLDER = PROJECT_ROOT / "data"
RINGS_FOLDER = "data/conf/RINGS"

# ===========================================
# ---------- EXTRACT PARAMETERS -------------
# ===========================================
IM_SIZE_ADJ = 512
RING_SIZE = 0.75

CANNY_TRHES1 = 50
CANNY_TRHES2 = 100

HCIRCLES_KERNEL_RATIO = 100
HCIRCLES_DP = 4
HCIRCLES_PAR1 = 50
HCIRCLES_PAR2 = 100
HCIRCLES_MINRAD = 127

HLINES_KERNEL_RATIO = 35

OCR_MINRATE = 0.65
OCR_N_READS = 4

N_HUMOMS = 7

KP_MAXCORNERS = 1000
KP_QUALITY = 0.1
KP_MINDIST = 5

NUM_LINES = 20

MIN_CENTERS_DIST = 500

SIFT_PERCENTAGE_FOR_GP = 0.7

LETTER_BASED_OCR = True
WORD_BASED_OCR = True
