import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, collate_fn
from .utils import load_data, save_model
import math
import argparse


def main(args):

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)

    train_data = load_data(args.train_path, tokenizer)
    valid_data = load_data(args.valid_path, tokenizer)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    config = {
        "d_model": 256,
        "d_head" : 32,
        "n_heads": 8,
        "n_layers": 4,
        "vocab_size": tokenizer.get_vocab_size(),
        "mode" : "standard",
        "tau": 1.5
    }

    model = LanguageModel(config)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    start_time = time.time()
    max_epochs = 100

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids[:, :-1], attention_mask[:, :-1]) 
            loss = criterion(outputs.view(-1, config["vocab_size"]), input_ids[:, 1:].reshape(-1))  # Target is next token
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        

        total_loss = 0.0
        total_chars = 0

        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids[:, :-1], attention_mask[:, :-1])
                loss = criterion(outputs.view(-1, config["vocab_size"]), input_ids[:, 1:].reshape(-1))

            total_loss += loss.item() * input_ids.size(0) * input_ids.size(1)  
            total_chars += input_ids.size(0) * input_ids.size(1)

        bpc = total_loss / (total_chars * math.log(2))
        print(f"Validation BPC: {bpc:.4f}")

        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Valid BPC: {bpc:.4f}")

        # Save the best model
        if bpc < best_loss:
            best_loss = bpc
            save_model(model, args.output_model_path)

        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > 23 * 900:
            print("Time limit exceeded. Stopping training.")
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
