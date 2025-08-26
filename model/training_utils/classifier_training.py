import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import glob
from sklearn.metrics import f1_score

def classification_training(
    embedder,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    checkpoint_interval: int, 
    path: str,
    continue_checkpoint: bool = False,
    writer: torch.utils.tensorboard.SummaryWriter | None = None,
    device: torch.device = None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    embedder.to(device)
    decoder.to(device)
    decoder.train()

    if not continue_checkpoint:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{decoder.__class__.__name__}_{timestamp}"
        log_dir = os.path.join(path, run_name)
        os.makedirs(log_dir, exist_ok=True)
        start_epoch = 1
    else:
        all_ckpts = glob.glob(os.path.join(path, f"{decoder.__class__.__name__}_*", "checkpoint_epoch_*.pt"))
        if not all_ckpts:
            raise FileNotFoundError("No checkpoint found to continue from.")
        last_ckpt = max(all_ckpts, key=os.path.getctime)
        log_dir = os.path.dirname(last_ckpt)
        checkpoint = torch.load(last_ckpt, map_location=device)
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from checkpoint {last_ckpt}, starting at epoch {start_epoch}")

    if writer is None:
        writer_dir = os.path.join(log_dir, "writer")
        os.makedirs(writer_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_dir)

    loss_fn = nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    for _ in range(start_epoch - 1):
        scheduler.step()

    for epoch in range(start_epoch, num_epochs + 1):
        running_loss = 0.0
        all_targets = []
        all_preds = []
        num_batches = len(train_loader)

        for batch_idx, (context, inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            targets = targets.float().to(device)  
            context_embed = embedder.embed(context).squeeze(0)
            inputs_embed = embedder.embed(inputs, return_tokens=True)

            optimizer.zero_grad()
            outputs = decoder(inputs_embed, context_embed).squeeze(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step = (epoch - 1) * num_batches + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

            all_targets.append(targets.detach().cpu())
            all_preds.append(outputs.detach().cpu())

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Train/LR", current_lr, epoch)

        all_targets = torch.cat(all_targets).numpy()
        all_preds = torch.cat(all_preds).numpy()
        pred_labels = (all_preds >= 0.5).astype(int)

        avg_loss = running_loss / num_batches
        accuracy = (pred_labels == all_targets).mean()
        f1 = f1_score(all_targets, pred_labels)

        writer.add_scalar("Train/Epoch_Avg_Loss", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy", accuracy, epoch)
        writer.add_scalar("Train/F1", f1, epoch)

        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")

        if (epoch % checkpoint_interval == 0) or (epoch == start_epoch):
            checkpoint_file = os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, checkpoint_file)
            print(f"Saved checkpoint: {checkpoint_file}")

    return avg_loss, writer
