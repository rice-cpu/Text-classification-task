# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import swanlab

from dataset import testDataset
from model import berttestClassifier

LR = 2e-5
BATCH_SIZE = 16
DROPOUT = 0.1
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    swanlab.login(api_key="wUnoyPsZavqKUxwFGpNwj")
    swanlab.init(
        project="Headline_Project_1",
        experiment_name=f"bert_lr{LR}_batch{BATCH_SIZE}",
        config={"lr": LR, "batch_size": BATCH_SIZE, "dropout": DROPOUT}
    )

    train_texts= []
    train_labels_raw= []

    with open("D:/PyCharm/01Text Classification/data/train_3k.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_!_")

            if len(parts) < 4:
                continue

            raw_label = parts[1]
            text = parts[3]

            train_texts.append(text)
            train_labels_raw.append(raw_label)

    val_texts = []
    val_labels_raw = []
    with open("D:/PyCharm/01Text Classification/data/dev_1k.txt", "r", encoding="utf-8") as f:
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

    NUM_CLASSES = len(unique_labels)
    print(f"数据清洗结束,检测到共有 {NUM_CLASSES} 个独特的分类标签，密码本映射为: {label_map}")



    # 组装数据
    train_loader = DataLoader(testDataset(train_texts, train_labels), batch_size=BATCH_SIZE, shuffle=True,num_workers=0)
    val_loader = DataLoader(testDataset(val_texts, val_labels), batch_size=4, shuffle=False,num_workers=0)

    # 初始化模型与工具
    model = berttestClassifier(num_classes=NUM_CLASSES, dropout_prob=DROPOUT)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 循环训练
    try:
        print("进入循环")
        model.train()
        epoch_loss = 0.0
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for batch_input_ids, batch_labels in train_loader:
                batch_input_ids = batch_input_ids.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_input_ids)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_input_ids, batch_labels in val_loader:
                    batch_input_ids = batch_input_ids.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_input_ids)
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())



            acc = accuracy_score(all_labels, all_preds)
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}] -> Loss: {avg_loss:.4f} | Acc: {acc:.2%}")
            scheduler.step()

            swanlab.log({"train_loss": avg_loss, "val_accuracy": acc, "epoch": epoch + 1})

    except Exception as e:
        print("\n这一行报错：")
        import traceback
        traceback.print_exc()