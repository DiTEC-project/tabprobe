import logging

import torch
import pandas as pd
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from src.initial_experiments.aerial_with_attention.aerial.data_preparation import _one_hot_encoding_with_feature_tracking

logger = logging.getLogger("aerial")


class AutoEncoder(nn.Module):
    def __init__(self, input_dimension, feature_count, layer_dims: list = None,
                 embed_dim: int = 64, num_heads: int = 4, attn_dropout: float = 0.1):
        super().__init__()

        self.input_dimension = input_dimension
        self.feature_count = feature_count

        # Determine layer dimensions
        if layer_dims is None:
            layer_count = max(1, math.ceil(math.log(input_dimension, 16)) - 1)
            reduction_ratio = (feature_count / input_dimension) ** (1 / (layer_count))
            dimensions = [input_dimension]
            for i in range(1, layer_count):
                next_dim = max(feature_count, int(dimensions[-1] * reduction_ratio))
                dimensions.append(next_dim)
            dimensions.append(feature_count)
        else:
            dimensions = [input_dimension] + layer_dims
        self.dimensions = dimensions

        # Encoder
        encoder_layers = []
        for i in range(len(dimensions) - 1):
            encoder_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i != len(dimensions) - 2:
                encoder_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_layers)

        latent_dim = dimensions[-1]

        # --- Column Attention ---
        self.attn_proj_in = nn.Linear(latent_dim, feature_count * embed_dim)
        self.column_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn_proj_mid = nn.Linear(feature_count * embed_dim, feature_count * embed_dim)

        # --- Row Attention ---
        self.row_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=False)
        self.attn_proj_out = nn.Linear(feature_count * embed_dim, latent_dim)

        # Decoder
        decoder_layers = []
        reversed_dimensions = list(reversed(dimensions))
        for i in range(len(reversed_dimensions) - 1):
            decoder_layers.append(nn.Linear(reversed_dimensions[i], reversed_dimensions[i + 1]))
            if i != len(reversed_dimensions) - 2:
                decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, feature_value_indices):
        # Encode
        y = self.encoder(x)  # (batch, latent_dim)

        # --- Column aerial_with_attention ---
        batch_size = y.size(0)
        proj = self.attn_proj_in(y)  # (batch, feature_count * embed_dim)
        proj = proj.view(batch_size, self.feature_count, -1)  # (batch, feature_count, embed_dim)

        col_out, _ = self.column_attention(proj, proj, proj)  # (batch, feature_count, embed_dim)

        # --- Row aerial_with_attention ---
        # swap: treat rows as sequence tokens
        row_in = col_out.permute(1, 0, 2)  # (feature_count, batch, embed_dim)
        row_out, _ = self.row_attention(row_in, row_in, row_in)  # (feature_count, batch, embed_dim)

        # back to (batch, feature_count, embed_dim)
        row_out = row_out.permute(1, 0, 2)

        # flatten and project back
        attn_flat = row_out.reshape(batch_size, -1)  # (batch, feature_count*embed_dim)
        y = self.attn_proj_out(attn_flat)  # (batch, latent_dim)

        # Decode
        y = self.decoder(y)

        # Feature-wise softmax
        chunks = [y[:, r.start:r.stop] for r in feature_value_indices]
        softmax_chunks = [F.softmax(chunk, dim=1) for chunk in chunks]
        y = torch.cat(softmax_chunks, dim=1)
        return y


def train(transactions: pd.DataFrame, autoencoder: AutoEncoder = None, noise_factor=0.5, lr=5e-3, epochs=2,
          batch_size=2, loss_function=torch.nn.BCELoss(), num_workers=1, layer_dims: list = None, device=None,
          patience: int = 20, delta: float = 1e-4):
    """
    train an autoencoder for association rule mining
    """
    input_vectors, feature_value_indices = _one_hot_encoding_with_feature_tracking(transactions, num_workers)

    if input_vectors is None:
        logger.error("Training stopped. Please fix the data issues first.")
        return None

    columns = input_vectors.columns.tolist()

    if not autoencoder:
        autoencoder = AutoEncoder(input_dimension=len(columns), feature_count=len(feature_value_indices),
                                  layer_dims=layer_dims)

    device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")
    autoencoder = autoencoder.to(device)
    autoencoder.train()
    autoencoder.input_vectors = input_vectors

    input_vectors = input_vectors.to_numpy(dtype=np.float32, copy=False)

    autoencoder.feature_value_indices = feature_value_indices
    autoencoder.feature_values = columns

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=2e-8)

    vectors_tensor = torch.from_numpy(input_vectors)

    dataset = TensorDataset(vectors_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())

    softmax_ranges = [range(cat['start'], cat['end']) for cat in feature_value_indices]

    best_loss = float("inf")
    patience_counter = 0

    total_batches = len(dataloader)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_index, (batch,) in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)

            noisy_batch = (batch + torch.randn_like(batch) * noise_factor).clamp(0, 1)
            reconstructed_batch = autoencoder(noisy_batch, softmax_ranges)

            total_loss = loss_function(reconstructed_batch, batch)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        epoch_loss /= total_batches

        # --- Early stopping ---
        if epoch_loss < best_loss - delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.debug(f"Early stopping triggered at epoch {epoch + 1}")
                break

    logger.debug("Training completed.")
    return autoencoder
