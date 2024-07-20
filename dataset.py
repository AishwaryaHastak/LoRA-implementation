"""
Load the emotion dataset from HuggingFace.
"""

from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
import torch

from constants import BATCH_SIZE, DATASET_URI, MODEL_NAME, TOKEN_MAX_LENGTH
from constants import TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET, SEED

class EmotionDataset():

    def __init__(self): 
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def __collate_data(self, batch):
        input_ids = torch.tensor([data['input_ids'] for data in batch])
        mask = torch.tensor([data['attention_mask'] for data in batch])
        labels = torch.tensor([data['label'] for data in batch])
        return input_ids, mask,labels
    
    def __tokenize(self,data):
        return self.tokenizer(data['text'], padding="max_length", truncation=True, max_length=TOKEN_MAX_LENGTH)
    
    def setup(self):
        print('\nDownloading the dataset...\n')
        dataset = load_dataset('dair-ai/emotion',download_mode='reuse_dataset_if_exists')

        print('\nTokenizing the dataset...\n')
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        tokenized_dataset = dataset.map(self.__tokenize, batched=True)

        train_dataset = tokenized_dataset['train'].shuffle(seed = SEED).select(range(TRAIN_SUBSET))
        val_dataset = tokenized_dataset['validation'].shuffle(seed = SEED).select(range(VAL_SUBSET))
        test_dataset = tokenized_dataset['test'].shuffle(seed = SEED).select(range(TEST_SUBSET))

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle= True, collate_fn= self.__collate_data)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn= self.__collate_data)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn= self.__collate_data)

        print('\nDataset ready...\n')
     
 

