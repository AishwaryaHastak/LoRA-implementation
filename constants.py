# Dataset parameters
DATASET_URI = "dair-ai/emotion"
TOKEN_MAX_LENGTH = 128
BATCH_SIZE = 10
TRAIN_SUBSET = 4000     # 16000
VAL_SUBSET = 1200       # 2000
TEST_SUBSET = 800       # 2000

# Dataset parameters
MODEL_NAME = 'distilbert-base-uncased'
NUM_LABELS = 6
RANK = 1
ALPHA = 1
LR = 5e-5
NUM_EPOCHS = 5
SEED = 42