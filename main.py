"""
Fine-tunes distill-bert large language model using LoRA
on a downstream task (IMDB sentiment classification).
"""

TOKEN_MAX_LENGTH = 32
BATCH_SIZE = 4
MODEL_NAME = 'distilbert-base-uncased'

