import os
import json
import torch
from transformers import BertTokenizer
from model import berttextClassifier


def inference():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = berttextClassifier(num_classes=15).to(device)

    checkpoint_path = "./checkpoints/best_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model, tokenizer, device, config


def predict(text, model, tokenizer, device):
    inputs = tokenizer(
        str(text),
        padding='max_length',
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs, dim=1).item()
    return prediction


if __name__ == "__main__":
    model, tokenizer, device, config = inference()
    test_file_path = os.path.join(config['data_dir'], "test_1k.txt")
    print(f"对数据集 {test_file_path}测验")

    fixed_class_list = [
        "100", "101", "102", "103", "104",
        "106", "107", "108", "109", "110",
        "112", "113", "114", "115", "116"
    ]

    correct = 0
    total = 0
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("_!_")
            try:
                raw_label = parts[1]
                title = parts[3]

                if len(parts) >= 5 and parts[4].strip():
                    text_content = title + " ， " + parts[4].strip()
                else:
                    text_content = title

                predicted_id = predict(text_content, model, tokenizer, device)
                true_id = fixed_class_list.index(raw_label)

                if predicted_id == true_id:
                    correct += 1

                total += 1
            except (IndexError, ValueError):
                continue

    print(f"测验结束。测试集共 {total} 条数据。")
    if total > 0:
        accuracy = correct / total
        print(f"模型预测正确数: {correct} 条")
        print(f"测试集最终准确率 (Accuracy): {accuracy:.4f}")