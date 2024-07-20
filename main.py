"""
Fine-tunes distill-bert large language model using LoRA.
"""

import os
import time

from dataset import EmotionDataset
from model import LoraModel
from constants import NUM_EPOCHS, LR

if __name__ == "__main__":
    # Get data loaders 
    dataset = EmotionDataset() 
    dataset.setup()

    lora_model = LoraModel()
    lora_model.train(dataset.train_loader, dataset.val_loader, lr = LR, num_epochs = NUM_EPOCHS )