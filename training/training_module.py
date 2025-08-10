# This module defines a PyTorch Lightning module that wraps the BERT classifier
# and provides training, validation, and testing logic for fine-tuning.

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from training.model_architecture import BertClassifier


class BertLitModule(pl.LightningModule):
    """
    LightningModule for fine-tuning a Transformer classifier on a multi-class task.

    Logs:
        - train_loss/train_acc per step and per epoch
        - val_loss/val_acc per epoch
    Also collects arrays for manual plotting at the end of training.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 4, lr: float = 2e-5):
        """
        Args:
            model_name (str): Hugging Face backbone to load.
            num_labels (int): Number of classes.
            lr (float): Learning rate for AdamW.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BertClassifier(model_name=model_name, num_labels=num_labels)
        self.lr = lr

        # Step-wise tracking (for custom plots)
        self.train_step_losses = []
        self.train_step_accs = []

        # Epoch-wise tracking (for custom plots)
        self.val_epoch_losses = []
        self.val_epoch_accs = []

        # Temporary per-batch accumulators for a single validation epoch
        self._val_loss_batches = []
        self._val_acc_batches = []

    # ---------- TRAIN ----------
    def training_step(self, batch, batch_idx):
        """
        One optimization step on a training batch.
        Returns the loss tensor used for the backward pass.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log to Lightning (progress bar + epoch aggregation)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc,  on_step=True, on_epoch=True, prog_bar=True)

        # Keep arrays for manual plots
        self.train_step_losses.append(loss.item())
        self.train_step_accs.append(acc.item())

        return loss

    # ---------- VALIDATION ----------
    def validation_step(self, batch, batch_idx):
        """
        One evaluation step on a validation batch.
        Aggregation across batches is handled in on_validation_epoch_end().
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Lightning averages these over the validation epoch
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc",  acc,  on_epoch=True, prog_bar=True)

        # Keep arrays for manual epoch plots
        self._val_loss_batches.append(loss.item())
        self._val_acc_batches.append(acc.item())

        return loss

    def on_validation_epoch_end(self):
        """
        Aggregate batch-level metrics collected during validation into epoch-level lists.
        """
        if self._val_loss_batches:
            self.val_epoch_losses.append(float(sum(self._val_loss_batches) / len(self._val_loss_batches)))
            self._val_loss_batches.clear()
        else:
            self.val_epoch_losses.append(float("nan"))

        if self._val_acc_batches:
            self.val_epoch_accs.append(float(sum(self._val_acc_batches) / len(self._val_acc_batches)))
            self._val_acc_batches.clear()
        else:
            self.val_epoch_accs.append(float("nan"))

    # ---------- OPTIMIZER ----------
    def configure_optimizers(self):
        """Configure and return the optimizer used for training."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # ---------- PLOTS ----------
    def on_train_end(self):
        """
        Save simple PNG plots (loss/accuracy) to the Lightning log directory.
        """
        import os
        import matplotlib.pyplot as plt

        log_dir = getattr(self.trainer, "log_dir", None)
        if not log_dir:
            return

        os.makedirs(log_dir, exist_ok=True)

        # Train loss (steps)
        if self.train_step_losses:
            plt.figure(figsize=(8, 4))
            plt.plot(self.train_step_losses, label="Train Loss")
            plt.title("Train Loss (steps)")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "train_loss_curve.png"))
            plt.close()

        # Train acc (steps)
        if self.train_step_accs:
            plt.figure(figsize=(8, 4))
            plt.plot(self.train_step_accs, label="Train Accuracy")
            plt.title("Train Accuracy (steps)")
            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "train_acc_curve.png"))
            plt.close()

        # Val loss (epochs)
        if self.val_epoch_losses:
            plt.figure(figsize=(8, 4))
            plt.plot(self.val_epoch_losses, marker="o", label="Val Loss")
            plt.title("Validation Loss (epochs)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "val_loss_curve.png"))
            plt.close()

        # Val acc (epochs)
        if self.val_epoch_accs:
            plt.figure(figsize=(8, 4))
            plt.plot(self.val_epoch_accs, marker="o", label="Val Accuracy")
            plt.title("Validation Accuracy (epochs)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "val_acc_curve.png"))
            plt.close()
