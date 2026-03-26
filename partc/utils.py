import os
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [tokenizer.encode(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx], dtype=torch.long),
            "attention_mask": torch.ones(len(self.data[idx]), dtype=torch.long)
        }

def load_data(file_path, tokenizer):
    return TextDataset(file_path, tokenizer)

def save_model(model, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pth'))