import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os


class TokenizeData:
    """
    Handles loading, splitting, tokenizing, and saving labeled conversational data
    for use with BERT-like models.
    """

    def __init__(self,
                 input_file: str,
                 output_dir: str = "processed_data",
                 pretrained_model: str = "distilbert-base-uncased",
                 max_length: int = 128):
        """
        Initialize the tokenizer pipeline.

        Args:
            input_file (str): Path to the labeled dataset (CSV).
            output_dir (str): Directory to save processed tensors.
            pretrained_model (str): Name of the Hugging Face model to use for tokenization.
            max_length (int): Maximum token length per input sequence.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.max_length = max_length

        os.makedirs(self.output_dir, exist_ok=True)
        self.data = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

    def load_labeled_data(self):
        """
        Load the labeled dataset from CSV and store it in self.data.
        Expects 'clean_conversation', 'product_label', and 'chat_id' columns.
        """
        self.data = pd.read_csv(self.input_file)
        print(f"Loaded dataset with {len(self.data)} rows.")

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into train and test subsets.

        Args:
            test_size (float): Proportion of data to use for testing.
            random_state (int): Seed for reproducibility.

        Returns:
            tuple: train_texts, test_texts, train_labels, test_labels, train_ids, test_ids
        """
        train_texts, test_texts, train_labels, test_labels, train_ids, test_ids = train_test_split(
            self.data['clean_conversation'],
            self.data['product_label'],
            self.data['chat_id'],
            test_size=test_size,
            stratify=self.data['product_label'],
            random_state=random_state
        )
        return (train_texts.tolist(), test_texts.tolist(),
                train_labels.tolist(), test_labels.tolist(),
                train_ids.tolist(), test_ids.tolist())

    def tokenize(self, texts):
        """
        Tokenize input texts using the specified BERT tokenizer.

        Args:
            texts (list): List of text strings.

        Returns:
            dict: Tokenized inputs with 'input_ids' and 'attention_mask'.
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def save_tensors(self, encodings, labels, ids, filename):
        """
        Save tokenized encodings, labels, and IDs as a PyTorch tensor file.

        Args:
            encodings (dict): Tokenized data.
            labels (list): Corresponding labels.
            ids (list): Corresponding record IDs.
            filename (str): Output file name.
        """
        dataset = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels),
            "ids": torch.tensor(ids)
        }
        torch.save(dataset, os.path.join(self.output_dir, filename))
        print(f"Saved {filename} with {len(labels)} samples.")

    def run(self):
        """
        Full pipeline:
        - Load labeled dataset
        - Split into train/test sets
        - Tokenize both sets
        - Save processed tensors to disk
        """
        self.load_labeled_data()
        train_texts, test_texts, train_labels, test_labels, train_ids, test_ids = self.split_data()

        print("Tokenizing training data...")
        train_encodings = self.tokenize(train_texts)

        print("Tokenizing test data...")
        test_encodings = self.tokenize(test_texts)

        self.save_tensors(train_encodings, train_labels, train_ids, "train_encodings.pt")
        self.save_tensors(test_encodings, test_labels, test_ids, "test_encodings.pt")

        print("Tokenization and saving completed successfully.")


if __name__ == "__main__":
    processor = TokenizeData(
        input_file="raw_data/synthetic_chat_dataset_labeled.csv"
    )
    processor.run()
