import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from training_module import BertLitModule
from dataset_module import ChatDataModule

# ------------------------------
# Data: prepare Lightning DataModule
# ------------------------------
data_module = ChatDataModule(batch_size=16)

# ------------------------------
# Model: instantiate LightningModule
# ------------------------------
model = BertLitModule(model_name="distilbert-base-uncased", num_labels=4, lr=2e-5)

# ------------------------------
# Callbacks: checkpoint + early stopping
# Note: both monitor "train_loss" in this setup
# ------------------------------
checkpoint_cb = ModelCheckpoint(
    monitor="train_loss",
    mode="min",
    save_top_k=1,
    filename="best_model"
)
early_stop_cb = EarlyStopping(
    monitor="train_loss",
    mode="min",
    patience=3
)

# ------------------------------
# Trainer: core training loop
# ------------------------------
trainer = Trainer(
    max_epochs=12,
    accelerator="auto",
    callbacks=[checkpoint_cb, early_stop_cb],
    log_every_n_steps=10
)

# ------------------------------
# Fit: start training
# ------------------------------
trainer.fit(model, datamodule=data_module)
