from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import os
import random
import torch.utils
import torch.utils.data

def trainer(device: str, model, optimizer, criterion, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, bert: torch.nn.Module, num_epochs: int = 3, save_path: str = "checkpoint"):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        for input_ids, attention_mask, labels in train_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = bert(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state  # (batch, seq_len, embed_dim)
            logits = model(embeddings)
            preds = torch.argmax(logits, dim=1)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        acc = correct / total
        print(f"Train Accuracy: {acc:.4f}")
        # 简单评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = bert(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                logits = model(embeddings)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Test Accuracy: {acc:.4f}")
    return model