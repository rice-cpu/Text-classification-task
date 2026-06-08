import os
import json
import torch
from transformers import BertTokenizer
from model import berttextClassifier


#初始化准备模型和分词器
def inference():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = berttextClassifier(num_classes=15).to(device)
    checkpoint_path = "./checkpoints/best_checkpoint.pt"

    model.eval()
    return model, tokenizer, device, config

#单句函数
def predict(text, model, tokenizer, device):

    inputs = tokenizer(
        str(text),
        padding='max_length',
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs, dim=1).item()
    return prediction

#测验
if __name__ == "__main__":
    model, tokenizer, device, config =inference()
    test_file_path = os.path.join(config['data_dir'], "test_1k.txt")
    print(f"对数据集 {test_file_path}测验")

    count = 0
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("_!_")
            if len(parts) < 4:
                continue

            raw_label = parts[1]
            text_content = parts[3]

            predicted_id = predict(text_content, model, tokenizer, device)
            count +=1

    print(f"测验结束。测试集共 {count} 条数据。")