import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import classification_report

from dataset import testDataset
from model import berttestClassifier

def run_inference():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    test_texts = []
    test_labels_raw = []
    test_file_path = os.path.join(config['data_dir'], "test_1k.txt")

    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("_!_")
            if len(parts) < 4: continue
            test_texts.append(parts[3])
            test_labels_raw.append(parts[1])

    unique_labels = sorted(list(set(test_labels_raw)))
    label_map = {raw_id: idx for idx, raw_id in enumerate(unique_labels)}
    test_labels = [label_map[raw_id] for raw_id in test_labels_raw]

    test_dataset = testDataset(test_texts, test_labels, config['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['dev_batch_size'], shuffle=False)

    checkpoint_path = "./checkpoints/best_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print(f"找不到已训练好的权重包 {checkpoint_path}，运行 main.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = berttestClassifier(num_classes=config['num_labels']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"还原该权重在实验第 {checkpoint['epoch']} 轮，验证集最高 Acc 为: {checkpoint['best_acc']:.2%}")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    target_names = [str(k) for k in label_map.keys()]


if __name__ == "__main__":
    run_inference()