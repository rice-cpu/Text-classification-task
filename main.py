import os
import json
import torch
import swanlab
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from model import berttestClassifier
from dataset import testDataset
from train import train_model

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def main():
    my_key = config.get("swanlab_api_key")
    if my_key:
        swanlab.login(api_key=my_key)
        print("SwanLab已登录")
    else:
        raise ValueError(
            "\n报错：config.json找不到名叫'swanlab_api_key'\n"
        )

    swanlab.init(
        project="Headline_Project_1",
        experiment_name=f"bert_lr{config['lr']}_batch{config['train_batch_size']}",
        config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = testDataset(
        file_path=os.path.join(config['data_dir'], "train_3k.txt"),
        max_len=config['max_length']
    )
    dev_dataset = testDataset(
        file_path=os.path.join(config['data_dir'], "dev_1k.txt"),
        max_len=config['max_length']
    )
    model = berttestClassifier(num_classes=train_dataset.num_classes).to(device)
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['dev_batch_size'], shuffle=False)

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