# This file defines the BertClassifier model using a pre-trained DistilBERT backbone
# and adds a classification head on top for downstream classification tasks.

import torch
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    """
    A simple classification head on top of a pre-trained Transformer encoder.

    Architecture:
        - Backbone: AutoModel.from_pretrained(model_name)
        - Head:     Linear(hidden_size -> num_labels)
    """

    def __init__(self, model_name="distilbert-base-uncased", num_labels=4):
        """
        Args:
            model_name (str): Hugging Face model id (e.g., 'distilbert-base-uncased').
            num_labels (int): Number of output classes.
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids (Tensor): Token ids, shape [B, L].
            attention_mask (Tensor): Attention mask, shape [B, L].

        Returns:
            Tensor: Logits, shape [B, num_labels].
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state           # [B, L, H]
        pooled_output = hidden_state[:, 0]                 # use first token (CLS-like) representation
        logits = self.classifier(pooled_output)            # [B, num_labels]
        return logits
