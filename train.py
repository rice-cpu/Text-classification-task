import os
import torch
import swanlab
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_model(model, train_loader, dev_loader, optimizer, scheduler, criterion, device, epochs):
    best_acc = 0.0
    SAVE_DIR = "./checkpoints"


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for batch in progress_bar:

            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)


            optimizer.zero_grad()

            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dev_loader:

                batch_input_ids = batch['input_ids'].to(device)
                batch_attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['label'].to(device)

                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        avg_loss = epoch_loss / len(train_loader)
        print(f"📈 Epoch [{epoch + 1}] 结束 -> 训练平均 Loss: {avg_loss:.4f} | 验证集 Acc: {acc:.2%}")


        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(SAVE_DIR, "best_checkpoint.pt")

            checkpoint_package = {
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_config': model.config.to_dict() if hasattr(model, 'config') else None
            }

            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(checkpoint_package, save_path)
            print(f"Checkpoint完整配置包已保存，最优模型: {save_path}")

        scheduler.step()
        swanlab.log({"train_loss": avg_loss, "val_accuracy": acc, "epoch": epoch + 1})

    print(f"\n训练结束，历史最佳准确率为: {best_acc:.2%}")