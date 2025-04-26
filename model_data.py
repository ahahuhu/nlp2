from huggingface_hub import Padding
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import os
import random
import torch.utils
import torch.utils.data

BERT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_cache")

class SentimentClass(torch.utils.data.Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = data
        self.label = label
    
    def __len__(self)-> int:
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        return self.data[index], self.label[index]

# 定义collate_fn，将文本批量编码为BERT词向量
def collate_fn(batch, tokenizer):
    texts, labels = zip(*batch)
    # tokenizer需提前定义好
    encodings = tokenizer(list(texts),
                         truncation=True,
                         padding=True,
                         max_length=128,
                         return_tensors='pt')
    # 返回input_ids, attention_mask, labels等
    return encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)

def get_data_loader(path_neg: str = "data/rt-polarity.neg", path_pos: str = "data/rt-polarity.pos", batch_size: int = 64, train_size: float = 0.8):
    """创建dataloader"""

    tokenizer = BertTokenizer.from_pretrained(BERT_CACHE_DIR, local_files_only=True)

    with open(path_neg) as f:
        data_neg = []
        for line in f.readlines():
            data_neg.append(line.strip())

    with open(path_pos) as f:
        data_pos = []
        for line in f.readlines():
            data_pos.append(line.strip())

    # 合并正负样本，创建标签
    all_data = data_pos + data_neg
    all_labels = [1] * len(data_pos) + [0] * len(data_neg)

    # 打乱数据
    combined = list(zip(all_data, all_labels))
    random.shuffle(combined)
    all_data, all_labels = zip(*combined)

    # 划分训练集和测试集
    split_idx = int(train_size * len(all_data))
    train_data, train_labels = list(all_data[:split_idx]), list(all_labels[:split_idx])
    test_data, test_labels = list(all_data[split_idx:]), list(all_labels[split_idx:])

    # 创建Dataset对象
    train_dataset = SentimentClass(train_data, train_labels)
    test_dataset = SentimentClass(test_data, test_labels)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 创建DataLoader时，加入collate_fn
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    pass