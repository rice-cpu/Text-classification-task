import os
import torch
import swanlab
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, dev_loader, optimizer, criterion, device, epochs):
    best_acc = 0.0
    SAVE_DIR = "./checkpoints"

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").long()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels").long()

                outputs = model(**batch)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch[{epoch + 1}]结束;训练Loss:{avg_loss:.4f};验证集Acc: {acc:.2%}")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(SAVE_DIR, "best_checkpoint.pt")
            model_config_dict = model.bert.config.to_dict()

            checkpoint_package = {
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config_dict
            }
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(checkpoint_package, save_path)

        swanlab.log({"train_loss": avg_loss, "val_accuracy": acc, "epoch": epoch + 1})

    print(f"训练结束，最高准确率为:{best_acc:.2%}")