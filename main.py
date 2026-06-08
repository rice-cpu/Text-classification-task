import os
import json
import torch
import swanlab
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer,DataCollatorWithPadding

from dataset import TextDataset
from model import berttextClassifier
from train import train_model

def main():
    CONFIG_PATH = "config.json"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    my_key = config.get("swanlab_api_key")
    if my_key:
        swanlab.login(api_key=my_key)
    else:
        raise ValueError("config.json里找不到名'swanlab_api_key'")

    swanlab.init(
        project="Headline_Project_1",
        experiment_name=f"bert_lr{config['learning_rate']}_batch{config['train_batch_size']}",
        config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    train_dataset = TextDataset(
        data_dir=config['data_dir'],
        filename="train_3k.txt",
        tokenizer=tokenizer,
        max_len=config['max_length']
    )

    dev_dataset = TextDataset(
        data_dir=config['data_dir'],
        filename="dev_1k.txt",
        tokenizer=tokenizer,
        max_len=config['max_length']
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=data_collator
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config['dev_batch_size'],
        shuffle=False,
        collate_fn=data_collator
    )
    model = berttextClassifier(num_classes=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 2e-5), eps=1e-8)

    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config['epochs']
    )
if __name__ == "__main__":
    main()