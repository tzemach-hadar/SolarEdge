import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle


class ChatDataset(Dataset):
    """
    Thin wrapper around a pre-tokenized .pt file produced by the preprocessing step.

    Expects the file (saved with torch.save) to contain a dict with:
        - "input_ids": Tensor [N, L]
        - "attention_mask": Tensor [N, L]
        - "labels": Tensor [N]
        - "ids": Tensor [N] (original record identifiers)
    """

    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to the .pt file with tokenized tensors.
        """
        self.data_path = data_path
        # Load the serialized tensors (CPU/GPU placement handled by PyTorch later)
        self.encodings = torch.load(self.data_path)

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        """
        Fetch a single sample by index.

        Returns:
            dict: A batch-ready dictionary with input_ids, attention_mask, labels, and ids.
        """
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['labels'][idx],
            'ids': self.encodings['ids'][idx]
        }


class ChatDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that provides train/val/test DataLoaders
    from pre-tokenized .pt files.
    """

    def __init__(self,
                 train_encodings_path="../data_preprocessing/processed_data/train_encodings.pt",
                 test_encodings_path="../data_preprocessing/processed_data/test_encodings.pt",
                 batch_size=16):
        """
        Args:
            train_encodings_path (str): Path to training set tensors.
            test_encodings_path  (str): Path to test/validation set tensors.
            batch_size (int): DataLoader batch size.
        """
        super().__init__()
        self.train_encodings_path = train_encodings_path
        self.test_encodings_path = test_encodings_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        """Instantiate Dataset objects for train and test/val phases."""
        self.train_dataset = ChatDataset(self.train_encodings_path)
        self.test_dataset = ChatDataset(self.test_encodings_path)

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return DataLoader for validation (uses test dataset here)."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
