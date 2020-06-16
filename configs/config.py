import os
import sys

sys.path.append("/usr/src/app/kaggle/trends-assessment-prediction")


TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 5

INPUT_DIR = 'inputs'
OUT_DIR = 'models'
FNC_PATH = os.path.join(INPUT_DIR, "fnc.csv")
LOADING_PATH = os.path.join(INPUT_DIR, "loading.csv")
TRAIN_SCORES_PATH = os.path.join(INPUT_DIR, "train_scores.csv")
TEST_MAP_PATH = INPUT_DIR + '/fMRI_test/'
TRAIN_MAP_PATH = INPUT_DIR + '/fMRI_train/'
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

FOLD0_ONLY = True
DEBUG = False
