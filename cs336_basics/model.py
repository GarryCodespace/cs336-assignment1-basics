import torch
import torch.nn as nn

from cs336_basics.model import Linear



# Convert text into token IDs
token_ids = tokenizer.encode("Hello world")

# Convert token IDs back into text
text = tokenizer.decode(token_ids)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # TODO 1: Create a weight matrix
        # Shape: (out_features, in_features)

        # TODO 2: Initialize the weights

    def forward(self, x):
        # TODO 3: Multiply x by the weight matrix
        pass