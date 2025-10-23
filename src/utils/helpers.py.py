# src/utils/helpers.py
import torch
import os

def save_checkpoint(model, optimizer, epoch, save_dir, best_acc):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc
    }
    torch.save(state, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
