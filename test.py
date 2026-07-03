import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
from model import berttextClassifier
from dataset import TextDataset


def test_evaluation():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get("model_name", "bert-base-chinese")
    num_classes = config.get("num_labels", 15)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = berttextClassifier(model_name=model_name, num_classes=num_classes).to(device)

    checkpoint_path = "./checkpoints/best_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    test_file_name ="test_1k.txt"
    test_dataset = TextDataset(
    data_dir=config['data_dir'],
    filename=test_file_name,
    tokenizer=tokenizer,
    config=config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('dev_batch_size', 32),
        shuffle=False,
        collate_fn = lambda x: TextDataset.collate_fn(x, tokenizer)
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="正在进行批量测试评估"):
            labels = batch.pop("labels").long()

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    total =len(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    correct =sum(1 for p, l in zip(all_preds, all_labels) if p == l)

    print("\n" + "=" * 30)
    print(f"测验结束,测试集共 {total} 条数据。")
    print(f"模型预测正确数: {correct} 条")
    print(f"测试集最终准确率(Accuracy): {acc:.4f} ({acc:.2%})")

if __name__ == "__main__":
    test_evaluation()