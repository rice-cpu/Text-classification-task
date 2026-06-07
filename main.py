import os
import json
import torch
import swanlab
from torch.utils.data import DataLoader
from openai import OpenAI
from transformers import BertTokenizer

from model import berttestClassifier
from dataset import testDataset
from train import train_model

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def main():
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    my_key = config.get("swanlab_api_key")

    if my_key:
        swanlab.login(api_key=my_key)
        print("SwanLab已登录")
    else:
        raise ValueError(
            "\n报错：config.json找不到名叫'swanlab_api_key'\n"
        )
    if deepseek_key:
        client = OpenAI(api_key=deepseek_key, base_url="https://api.api-deepseek.com")

    swanlab.init(
        project="Headline_Project_1",
        experiment_name=f"bert_lr{config['lr']}_batch{config['train_batch_size']}",
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


    train_texts = []
    train_labels_raw = []
    train_file = os.path.join(config['data_dir'], "train_3k.txt")
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_!_")
            if len(parts) < 4:
                continue
            train_texts.append(parts[3])
            train_labels_raw.append(parts[1])

    val_texts = []
    val_labels_raw = []
    dev_file = os.path.join(config['data_dir'], "dev_1k.txt")
    with open(dev_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_!_")
            if len(parts) < 4:
                continue
            val_texts.append(parts[3])
            val_labels_raw.append(parts[1])


    all_raw_labels = train_labels_raw + val_labels_raw
    unique_labels = sorted(list(set(all_raw_labels)))
    label_map = {raw_id: idx for idx, raw_id in enumerate(unique_labels)}

    train_labels = [label_map[raw_id] for raw_id in train_labels_raw]
    val_labels = [label_map[raw_id] for raw_id in val_labels_raw]

    print(f"共有 {len(unique_labels)} 个类别")

    train_dataset = testDataset(
        texts=train_texts,
        labels=train_labels,
        max_len=config['max_length']
    )

    dev_dataset = testDataset(
        texts=val_texts,
        labels=val_labels,
        max_len=config['max_length']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['dev_batch_size'], shuffle=False)

    model = berttestClassifier(num_classes=config['num_labels']).to(device)
    optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters() if "bert" in n],
            "lr": 5e-6, },
        {
            "params": [p for n, p in model.named_parameters() if "bert" not in n],
            "lr": 5e-4,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'])

    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=config['epochs']
    )

if __name__ == "__main__":
    main()