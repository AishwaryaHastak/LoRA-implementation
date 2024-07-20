# Dataset parameters
DATASET_URI = "dair-ai/emotion"
TOKEN_MAX_LENGTH = 128
BATCH_SIZE = 10
TRAIN_SUBSET = 4000     # 16000
VAL_SUBSET = 1200       # 2000
TEST_SUBSET = 800       # 2000

MODEL_NAME = 'distilbert-base-uncased'
RANK = 1
ALPHA = 1
LR = 1e-4
NUM_EPOCHS = 4