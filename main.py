import os
import json
import torch
import swanlab
import torch.nn as nn
import random
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer

from dataset import TextDataset
from model import berttextClassifier
from train import train_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def my_collate_fn(batch):

    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_batch_len = max(len(ids) for ids in input_ids)


    padded_input_ids = []
    padded_attention_masks = []
    padded_token_type_ids = []

    for ids, mask,tti in zip(input_ids, attention_masks,token_type_ids):
        pad_len = max_batch_len - len(ids)
        padded_input_ids.append(ids + [0] * pad_len)
        padded_attention_masks.append(mask + [0] * pad_len)
        padded_token_type_ids.append(tti+ [0] * pad_len)

    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
        'token_type_ids': torch.tensor(padded_token_type_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }


def main():
    CONFIG_PATH = "config.json"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    my_key = config.get("swanlab_api_key")
    if my_key:
        swanlab.login(api_key=my_key)
    else:
        raise ValueError("config.json里找不到'swanlab_api_key'")

    swanlab.init(
        project="Headline_Project_1",
        experiment_name=f"bert_lr{config['learning_rate']}_batch{config['train_batch_size']}",
        config=config
    )
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get("model_name", "bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = TextDataset(
        data_dir=config['data_dir'],
        filename="train_3k.txt",
        tokenizer=tokenizer,
        config=config
    )

    dev_dataset = TextDataset(
        data_dir=config['data_dir'],
        filename="dev_1k.txt",
        tokenizer=tokenizer,
        config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=my_collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config['dev_batch_size'],
        shuffle=False,
        collate_fn=my_collate_fn
    )

    num_classes = config.get('num_labels', 15)
    model = berttextClassifier(model_name=model_name, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 2e-5), eps=1e-8)

    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config['epochs'],
        patience=config.get('patience', 3)
    )


if __name__ == "__main__":
    main()