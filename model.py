"""
Define LoRA Model and apply LoRA to linear layers of the attention heads.
"""

import torch
import torch.nn as nn
from constants import RANK, ALPHA, MODEL_NAME

from tqdm import tqdm
from torch.optim import Adam 
from torch.optim.lr_scheduler import ExponentialLR,ReduceLROnPlateau
from transformers import DistilBertForSequenceClassification


class LoraLayer(nn.Module):
    def __init__(self, lin_layer, rank, alpha):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_feature,out_feature = lin_layer.in_features, lin_layer.out_features
        self.A = nn.Parameter(torch.zeros(in_feature, rank)).to(self.device)
        nn.init.normal_(self.A, mean=0, std=1)
        self.B = nn.Parameter(torch.zeros(rank, out_feature)).to(self.device)
        self.scale = alpha / rank
        self.W = lin_layer.to(self.device)

    def forward(self,x):
        return self.W(x) + self.scale * (torch.matmul(torch.matmul(x, self.A), self.B))
    

class LoraModel():
    def __init__(self,  rank = RANK, alpha = ALPHA, apply_lora=True):
        self.base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.base_model.to(self.device)

        for name,param in self.base_model.named_parameters():
            if 'attention' in name:
                param.requires_grad = False 

        if apply_lora:
            self.__apply_lora(rank, alpha)

    def __apply_lora(self, rank, alpha):
        for block in self.base_model.distilbert.transformer.layer: 
            block.attention.q_lin = LoraLayer(block.attention.q_lin, rank, alpha)
            block.attention.v_lin = LoraLayer(block.attention.v_lin, rank, alpha)
            block.attention.k_lin = LoraLayer(block.attention.k_lin, rank, alpha)
            block.attention.out_lin = LoraLayer(block.attention.out_lin, rank, alpha)

    def train(self, train_loader, val_loader, lr=1e-5, num_epochs=10):
        optimizer = Adam(self.base_model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        
        self.base_model.train()
        
        
        for i in range(num_epochs):
            correct_pred, train_loss, total_train = 0, 0, 0  
            
            for step, batch in enumerate(tqdm(train_loader)): 
                input_ids, mask, labels = batch 
                input_ids, mask, labels = input_ids.to(self.device), mask.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.base_model(input_ids, attention_mask=mask, labels=labels)
 
                loss = outputs.loss
                loss.backward()
                optimizer.step() 

                predictions = torch.max(outputs.logits, dim=-1).indices 
                correct_pred += (predictions == labels).sum().item()
                train_loss += loss.item()
                total_train += len(labels) 

                # Update progress bar every 40 steps
                if step % 100 == 0: 
                    # print(f'correct {correct_pred} total {total_train}')
                    print('Step:', step, '\n [TRAIN] Loss:', train_loss/total_train, 'accuracy:', correct_pred/total_train)            

            self.save_model()

            train_loss_avg = train_loss / total_train
            train_acc_avg = correct_pred / total_train
            # print(f'correct {correct_pred} total {total_train}')
            print(f'[TRAIN] Epoch {i+1} Loss: {train_loss_avg}, Accuracy: {train_acc_avg}')

            val_loss_avg, val_acc_avg = self.predict(val_loader)
            print(f'[VAL] Epoch {i+1} Loss: {val_loss_avg}, Accuracy: {val_acc_avg}')

            scheduler.step(val_loss_avg)

    def predict(self, data_loader):
        self.base_model.eval()
        correct_pred, val_test_loss, total_data = 0, 0, 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids, mask, labels = batch 
                input_ids, mask, labels = input_ids.to(self.device), mask.to(self.device), labels.to(self.device)
                
                output = self.base_model(input_ids, attention_mask=mask, labels=labels)
                val_test_loss += output.loss.item() 
                prediction = torch.argmax(output.logits, dim=-1) 
                correct_pred += (prediction == labels).sum().item()
                total_data += len(labels)

        avg_loss = val_test_loss / total_data
        avg_acc = correct_pred / total_data
        return avg_loss, avg_acc

    def save_model(self, epoch):
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_path = os.path.join(output_dir, f"model.pt")
        torch.save(self.base_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")