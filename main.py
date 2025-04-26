import torch
import torch.utils
import trainer
from model.cnn_model import CNNModel
from model.lstm_model import LSTMModel
from model_data import get_data_loader
from torch import dropout, nn
import os, argparse
from argparse import Namespace

from transformers import BertModel, BertTokenizer
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

BERT_CACHE_DIR = "bert_cache"

def get_bert_and_tokenizer():
    # 如果本地没有缓存，则下载并保存
    if not os.path.exists(BERT_CACHE_DIR):
        os.makedirs(BERT_CACHE_DIR, exist_ok=True)
        BertModel.from_pretrained('bert-base-uncased').save_pretrained(BERT_CACHE_DIR)
        BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained(BERT_CACHE_DIR)
    # 从本地加载
    bert = BertModel.from_pretrained(BERT_CACHE_DIR).to(device)
    tokenizer = BertTokenizer.from_pretrained(BERT_CACHE_DIR)
    return bert, tokenizer

def get_arguments():
    """对数据集的位置，模型和训练的配置进行调整"""
    args = argparse.ArgumentParser()
    args.add_argument("model_name", type=str)
    args.add_argument("--num_layers", type=int, default=1)
    args.add_argument("--dropout", type=float, default=0.2)
    args.add_argument("--bidirectional", action="store_true")
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--lr", type=float, default=2e-5)
    args.add_argument("--data_neg", type=str, default="data/rt-polarity.neg")
    args.add_argument("--data_pos", type=str, default="data/rt-polarity.pos")
    args.add_argument("--mode", type=str, default="train")
    return args.parse_args()

def test_model(args: Namespace):
    model_path = f"checkpoint/{args.model_name}-{args.epochs}-{args.lr:f}-{args.dropout}-{args.num_layers}-{args.bidirectional}.pt"
    if not os.path.isfile(model_path):
        raise FileNotFoundError("模型位置有误或模型参数错误")
    if args.model_name == "CNN":
        model = CNNModel(conv_layers=args.num_layers, dropout=args.dropout)
    # TODO 加上RNN这里
    model = model.to(device=device)
    with open(model_path, mode="rb") as f:
        buffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(buffer))
    bert, tokenizer = get_bert_and_tokenizer()
    model.eval()
    sentence = input("请输入要测试的句子, 输入为空则自动退出\n")
    while sentence:
        # 将单句包装成列表，获得batch形式的编码
        inputs = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            embeddings = outputs.last_hidden_state  # (batch, seq_len, embed_dim)
            prod: torch.Tensor = model(embeddings)
        print("<=========>")

        print(prod, type(prod))
        print(f"输入句子为: {sentence}\n分数为{prod},初步判断为{'positive' if prod[0][1] >= prod[0][0] else 'negetive'}")
        print("<============>")
        sentence = input("请输入要测试的句子, 输入为空则自动退出\n")


def train(args: Namespace):
    train_dataloader, test_dataloader = get_data_loader(args.data_neg, args.data_pos)
    model = None
    if args.model_name == "CNN":
        model = CNNModel(conv_layers=args.num_layers, dropout=args.dropout)
    elif args.model_name == "LSTM":
        model = LSTMModel(num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    bert, tokenizer = get_bert_and_tokenizer()
    trained_model = trainer.trainer(device, model, optimizer, criterion, train_dataloader, test_dataloader, bert, args.epochs)
    save_path = "checkpoint"

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    save_path = f"{save_path}/{args.model_name}-{args.epochs}-{args.lr:f}-{args.dropout}-{args.num_layers}-{args.bidirectional}.pt"
    # 修正保存模型的方法名和文件名
    torch.save(trained_model.state_dict(), save_path)
    print(f'模型已成功保存到{save_path}')

        
    
    

if __name__ == "__main__":
    args = get_arguments()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test_model(args)

