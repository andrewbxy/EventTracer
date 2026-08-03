import argparse
import os
import random
from loguru import logger as guru
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from spikingjelly.activation_based import functional

from dataloader import EventDataset
from model import EvSNet
from utils import emd, abs_l1

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--expname",
    type=str,
    default=None,
    required=True,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train the model",
)
parser.add_argument(
    "--latent_dim",
    type=int,
    default=64,
    help="Latent dimension of the model",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Learning rate"
)
parser.add_argument(
    "--val_iters",
    type=int,
    default=10000000
)
parser.add_argument(
    "--val_epochs",
    type=int,
    default=4
)
args = parser.parse_args()

log_dir = os.path.join("logs", args.expname)
train_figs_dir = os.path.join(log_dir, "train")
val_figs_dir = os.path.join(log_dir, "val")
checkpoints_dir = os.path.join(log_dir, "checkpoints")
onnx_dir = os.path.join(log_dir, "onnx")
os.makedirs(train_figs_dir, exist_ok=True)
os.makedirs(val_figs_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(onnx_dir, exist_ok=True)

positions = [(row, col) for row in range(20) for col in range(12)]
random.shuffle(positions)

train_positions = positions[:int(len(positions) * 0.8)]
val_positions = positions[int(len(positions) * 0.8):]

data_dirs = ["Bistro-Exterior-Day", "Bistro-Exterior-Night", "Bistro-Interior", "classroom", "kitchen", "staircase"]
max_lens = [1984, 1984, 1984, 3968, 3968, 3968]

train_datasets = []
for data_dir, max_len in zip(data_dirs, max_lens):
    train_datasets.append(
        EventDataset(
            data_dir,
            train_positions,
            max_len,
            1000,
            True,
            type='log',
        )
    )
train_dataset = ConcatDataset(train_datasets)

val_datasets = []
for data_dir, max_len in zip(data_dirs, max_lens):
    val_datasets.append(
        EventDataset(
            data_dir,
            val_positions,
            max_len,
            1000,
            True,
            type='log',
            train=False,
        )
    )
val_dataset = ConcatDataset(val_datasets)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=None,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=None,
)
guru.info(f"{len(train_loader)} batches in train set")
guru.info(f"{len(val_loader)} batches in val set")

model = EvSNet(hidden_channels=args.latent_dim)
model = model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

best_val_loss = float("inf")
best_epoch = 0
best_iter = 0
train_count = 0

dummy_input = torch.randn(4096, 1, 43, device='cuda')

for epoch in range(1, args.epochs + 1):
    model.train()
    loss_epoch = 0
    pbar = tqdm(train_loader)
    for x_seq_sample, y_seq_sample in pbar:
        functional.reset_net(model)
        x_seq_sample = x_seq_sample.to('cuda', non_blocking=True)[0]
        y_seq_sample = y_seq_sample.to('cuda', non_blocking=True)[0]
        
        x = (x_seq_sample[1:] - x_seq_sample[:-1]).permute(1, 2, 0)
        y = y_seq_sample[1:].permute(1, 2, 0)

        pred = model(x)
        y_processed = torch.cat(((y > 0).float(), (y < 0).float()), dim=1)
        x_processed = torch.cat(((pred > 0).float() * pred, (pred < 0).float() * (-pred)), dim=1)
        loss = emd(x_processed, y_processed) + abs_l1(x_processed, y_processed) # [T, B, 1]
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pbar.set_postfix({"loss": loss.item()})
        loss_epoch += loss.item()

        train_count += 1
        scheduler.step()
        
        if train_count % args.val_iters == 0:
            val_pbar = tqdm(val_loader, desc=f"Validation Iteration {train_count}", leave=False)
            val_epoch_loss = 0
            val_count = 0
            model.eval()
            with torch.no_grad():
                for val_x_seq_sample, val_y_seq_sample in val_pbar:
                    functional.reset_net(model)
                    val_x_seq_sample = val_x_seq_sample.to('cuda', non_blocking=True)[0]
                    val_y_seq_sample = val_y_seq_sample.to('cuda', non_blocking=True)[0]

                    val_x = (val_x_seq_sample[1:] - val_x_seq_sample[:-1]).permute(1, 2, 0)
                    val_y = val_y_seq_sample[1:].permute(1, 2, 0)
                    val_pred = model(val_x)  # [T, B, 1]
                    val_y_processed = torch.cat(((val_y > 0).float(), (val_y < 0).float()), dim=1)
                    val_x_processed = torch.cat(((val_pred > 0).float(), (val_pred < 0).float()), dim=1)
                    val_loss = emd(val_x_processed, val_y_processed)
                    val_pbar.set_postfix({"val_loss": val_loss.item()})
                    val_epoch_loss += val_loss.item()
                    val_count += 1

            model.train()
            val_epoch_loss /= len(val_loader)
            guru.info(f"Iteration {train_count}, Val loss: {val_epoch_loss:.4f}")

            if val_epoch_loss < best_val_loss:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_iter = train_count
                    
                checkpoint_path = os.path.join(checkpoints_dir, f"best_model_iter_{train_count}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "iter": train_count,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                    },
                    checkpoint_path,
                )
                # Export the best model to ONNX format
                torch.onnx.export(
                    model,
                    dummy_input,
                    os.path.join(onnx_dir, f"best_model_iter_{train_count}.onnx"),
                    input_names=["input"],
                    output_names=["output"],
                    do_constant_folding=True,
                    dynamic_axes=None,
                    keep_initializers_as_inputs=False,
                    export_params=True,
                    verbose=False,
                    dynamo=True,
                    report=True,
                    optimize=True,
                    verify=True,
                    fallback=True,
                    external_data=False,
                )
                guru.info(
                    f"Saved best model checkpoint at iteration {train_count} with val_loss: {best_val_loss:.4f} to {checkpoint_path}"
                )

    loss_epoch /= len(train_loader)
    guru.info(f"Epoch {epoch}, Train loss: {loss_epoch:.4f}")

    if epoch % args.val_epochs == 0 or epoch == args.epochs:
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False)
        val_epoch_loss = 0
        val_count = 0
        model.eval()
        with torch.no_grad():
            for val_x_seq_sample, val_y_seq_sample in val_pbar:
                functional.reset_net(model)
                val_x_seq_sample = val_x_seq_sample.to('cuda', non_blocking=True)[0]
                val_y_seq_sample = val_y_seq_sample.to('cuda', non_blocking=True)[0]

                val_x = (val_x_seq_sample[1:] - val_x_seq_sample[:-1]).permute(1, 2, 0)
                val_y = val_y_seq_sample[1:].permute(1, 2, 0)
                val_pred = model(val_x)  # [T, B, 1]
                val_y_processed = torch.cat(((val_y > 0).float(), (val_y < 0).float()), dim=1)
                val_x_processed = torch.cat(((val_pred > 0).float(), (val_pred < 0).float()), dim=1)
                val_loss = emd(val_x_processed, val_y_processed)
                val_pbar.set_postfix({"val_loss": val_loss.item()})
                val_epoch_loss += val_loss.item()

                val_count += 1

            val_epoch_loss /= len(val_loader)
            guru.info(f"Epoch {epoch}, Val loss: {val_epoch_loss:.4f}")

            if val_epoch_loss < best_val_loss or epoch % 10 == 0:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_epoch = epoch
                    
                checkpoint_path = os.path.join(checkpoints_dir, f"best_model_epoch_{epoch}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "iter": train_count,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                    },
                    checkpoint_path,
                )
                # Export the best model to ONNX format
                torch.onnx.export(
                    model,
                    dummy_input,
                    os.path.join(onnx_dir, f"best_model_epoch_{epoch}.onnx"),
                    input_names=["input"],
                    output_names=["output"],
                    do_constant_folding=True,
                    dynamic_axes=None,
                    keep_initializers_as_inputs=False,
                    export_params=True,
                    verbose=False,
                    dynamo=True,
                    report=True,
                    optimize=True,
                    verify=True,
                    fallback=True,
                    external_data=False,
                )
                guru.info(
                    f"Saved best model checkpoint at epoch {epoch} with val_loss: {best_val_loss:.4f} to {checkpoint_path}"
                )

guru.info(
    f"Training finished. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}"
)
# Save the final model as well
final_checkpoint_path = os.path.join(checkpoints_dir, "final_model.pth")
torch.save(
    {
        "epoch": args.epochs,
        "iter": train_count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_epoch,  # Storing last training loss here
        "best_val_loss_overall": best_val_loss,
        "best_val_epoch_overall": best_epoch,
        "best_val_iter_overall": best_iter,
    },
    final_checkpoint_path,
)
# Export the final model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    os.path.join(onnx_dir, "final_model.onnx"),
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    dynamic_axes=None,
    keep_initializers_as_inputs=False,
    export_params=True,
    verbose=False,
    dynamo=True,
    report=True,
    optimize=True,
    verify=True,
    fallback=True,
    external_data=False,
)
guru.info(f"Saved final model checkpoint to {final_checkpoint_path}")
