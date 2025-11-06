import torch
import torch.nn as nn
import numpy as np
from itertools import combinations, product


class RowEncoder(nn.Module):
    """Encodes a single row (sample) from input dimension to embedding dimension"""

    def __init__(self, input_dim, embed_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class AutoEncoder(nn.Module):
    """
    PyAerial Autoencoder - Following the exact architecture from PyAerial
    Automatically determines layer structure based on input dimension and feature count
    """

    def __init__(self, input_dimension, feature_count):
        """
        :param input_dimension: Number of features after one-hot encoding
        :param feature_count: Target feature count (bottleneck dimension)
        """
        super().__init__()

        self.input_dimension = input_dimension
        self.feature_count = feature_count

        # Fixed architecture: 2 hidden layers with 1/4 reduction each
        hidden_dim1 = max(feature_count, input_dimension // 2)
        hidden_dim2 = max(feature_count, hidden_dim1 // 2)

        # Dimensions: input -> hidden1 -> hidden2 -> bottleneck
        self.dimensions = [input_dimension, hidden_dim1, hidden_dim2, feature_count]

        # Build Encoder: input -> hidden1 -> hidden2 -> bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, feature_count)
        )

        # Build Decoder (mirror): bottleneck -> hidden2 -> hidden1 -> input
        self.decoder = nn.Sequential(
            nn.Linear(feature_count, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, input_dimension)
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """Xavier initialization as in PyAerial"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def forward(self, x, feature_value_indices):
        """
        Forward pass with softmax normalization per feature (PyAerial style)
        :param x: Input tensor
        :param feature_value_indices: List of ranges for softmax application
        """
        y = self.encoder(x)
        y = self.decoder(y)

        # Apply softmax to each feature range (PyAerial's key innovation)
        chunks = [y[:, range_obj.start:range_obj.stop] for range_obj in feature_value_indices]
        softmax_chunks = [torch.softmax(chunk, dim=1) for chunk in chunks]
        y = torch.cat(softmax_chunks, dim=1)

        return y


class TabularFoundationModel(nn.Module):
    """
    Tabular Foundation Model with PyAerial Autoencoder as query

    TRUE FOUNDATION MODEL: Handles tables with varying numbers of rows and columns
    through padding and masking, with fixed-size architecture.

    Architecture Flow:
    1. Row Encoding: Each row is encoded using a SHARED encoder
       - Handles variable input dimensions through padding
       - Much more efficient than having separate encoders per row

    2. Column-wise Attention: Features (embed_dim dimensions) attend to other features
       - FIXED architecture independent of number of rows
       - Uses embed_dim x embed_dim aerial_with_attention (not row-dependent)

    3. Cross-Attention: Query (code layer) attends to the processed table
       - The autoencoder's code layer uses the table context for reconstruction
       - Uses masking to handle variable number of rows

    Args:
        input_dim: Maximum input feature dimension (d) for table rows
        embed_dim: Embedding dimension (ed) for table rows
        ae_input_dim: Input dimension for the autoencoder (query)
        ae_feature_count: Number of features for autoencoder bottleneck
        max_rows: Maximum number of rows supported (default 1000)
        num_heads: Number of aerial_with_attention heads
        num_layers: Number of column aerial_with_attention layers
        dropout: Dropout rate
    """

    def __init__(self, input_dim, embed_dim, ae_input_dim, ae_feature_count,
                 max_rows=1000, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.ae_input_dim = ae_input_dim
        self.ae_feature_count = ae_feature_count
        self.max_rows = max_rows

        # Row encoder: SHARED encoder applied to each row
        self.row_encoder = RowEncoder(input_dim, embed_dim)

        # Simplified Autoencoder with fixed 2 hidden layers
        self.autoencoder = AutoEncoder(ae_input_dim, ae_feature_count)

        # FIXED Column-wise Attention (independent of n_rows)
        # Features (embed_dim) attend to other features
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Multi-head aerial_with_attention components for column aerial_with_attention
        # These operate on embed_dim dimension (features attending to features)
        self.col_query = nn.Linear(embed_dim, embed_dim)
        self.col_key = nn.Linear(embed_dim, embed_dim)
        self.col_value = nn.Linear(embed_dim, embed_dim)
        self.col_out = nn.Linear(embed_dim, embed_dim)

        # Feed-forward network for column aerial_with_attention
        self.col_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.col_norm1 = nn.LayerNorm(embed_dim)
        self.col_norm2 = nn.LayerNorm(embed_dim)

        self.dropout = dropout
        self.num_col_layers = num_layers

        # Cross-aerial_with_attention: code layer (query) attends to embedded table rows
        self.ae_code_dim = self.autoencoder.dimensions[-1]

        # Adjust num_heads to be compatible with ae_code_dim
        cross_attn_heads = min(num_heads, self.ae_code_dim)
        while self.ae_code_dim % cross_attn_heads != 0 and cross_attn_heads > 1:
            cross_attn_heads -= 1

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.ae_code_dim,
            num_heads=cross_attn_heads,
            dropout=dropout,
            batch_first=True
        )

        # Project embedded rows to match code dimension for cross-aerial_with_attention
        self.query_to_table_attention = nn.Linear(embed_dim, self.ae_code_dim)

    def column_attention_layer(self, x, row_mask=None):
        """
        FIXED column aerial_with_attention: features attend to features.
        Architecture is independent of n_rows - operates on embed_dim.

        Args:
            x: (batch_size, n_rows, embed_dim)
            row_mask: Optional mask (batch_size, n_rows) for padding
        Returns:
            attended: (batch_size, n_rows, embed_dim)
        """
        batch_size, n_rows, embed_dim = x.shape

        # Multi-head aerial_with_attention where each embedding dimension attends across rows
        # This is row-independent: each feature learns patterns across the row dimension

        # Project to Q, K, V
        Q = self.col_query(x)  # (B, n_rows, embed_dim)
        K = self.col_key(x)  # (B, n_rows, embed_dim)
        V = self.col_value(x)  # (B, n_rows, embed_dim)

        # Reshape for multi-head aerial_with_attention
        Q = Q.view(batch_size, n_rows, self.num_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, n_rows, head_dim)
        K = K.view(batch_size, n_rows, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_rows, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product aerial_with_attention (rows attend to rows, but weights are shared across all tables)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, n_rows, n_rows)

        # Apply row mask if provided (mask out padding in KEY dimension)
        if row_mask is not None:
            # row_mask shape: (B, n_rows), True = keep, False = mask
            # Expand to (B, 1, 1, n_rows) to broadcast over (B, n_heads, n_rows_query, n_rows_key)
            # This masks the KEY dimension (last dimension)
            mask_expanded = row_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, n_rows)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # Softmax over KEY dimension (last dim)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply aerial_with_attention
        attended = torch.matmul(attn_weights, V)  # (B, n_heads, n_rows, head_dim)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, n_rows, embed_dim)

        # Output projection
        attended = self.col_out(attended)
        attended = torch.nn.functional.dropout(attended, p=self.dropout, training=self.training)

        # Residual connection + Layer norm
        x = self.col_norm1(x + attended)

        # Feed-forward network
        ffn_out = self.col_ffn(x)

        # Residual connection + Layer norm
        x = self.col_norm2(x + ffn_out)

        return x

    def forward(self, table_rows, query_data, feature_value_indices, table_mask=None):
        """
        Forward pass with support for variable-sized tables through padding/masking.

        Args:
            table_rows: Table data (batch_size, n_rows, input_dim) where n_rows <= max_rows
            query_data: Query data for autoencoder (batch_size, n_query, ae_input_dim)
            feature_value_indices: List of ranges for softmax application
            table_mask: Optional mask for table rows (batch_size, n_rows), True = valid row

        Returns:
            reconstructed: Reconstructed query data (batch_size, n_query, ae_input_dim)
            attended_code: Code representations after attending to table (batch_size, n_query, ae_code_dim)
        """
        batch_size, n_rows, feat_dim = table_rows.shape

        # Check max_rows constraint
        assert n_rows <= self.max_rows, f"n_rows ({n_rows}) exceeds max_rows ({self.max_rows})"

        # Zero padding for table rows if needed (handle variable column count)
        if feat_dim < self.input_dim:
            padding = torch.zeros(batch_size, n_rows, self.input_dim - feat_dim,
                                  device=table_rows.device, dtype=table_rows.dtype)
            table_rows = torch.cat([table_rows, padding], dim=-1)

        # Encode each row: (batch_size, n_rows, input_dim) -> (batch_size, n_rows, embed_dim)
        embedded_rows = self.row_encoder(table_rows)

        # Check for NaN after encoding
        if torch.isnan(embedded_rows).any():
            print(f"WARNING: NaN detected after row encoding!")

        # Apply FIXED column aerial_with_attention (independent of n_rows, multiple layers)
        # Use masking to handle variable number of rows
        for layer_idx in range(self.num_col_layers):
            embedded_rows = self.column_attention_layer(embedded_rows, row_mask=table_mask)
            if torch.isnan(embedded_rows).any():
                print(f"WARNING: NaN detected after column aerial_with_attention layer {layer_idx}!")

        # Handle query data dimensions
        if query_data.dim() == 2:
            query_data = query_data.unsqueeze(1)

        query_dim = query_data.shape[2]

        # Zero padding for query data if needed
        if query_dim < self.ae_input_dim:
            padding = torch.zeros(batch_size, query_data.shape[1], self.ae_input_dim - query_dim,
                                  device=query_data.device, dtype=query_data.dtype)
            query_data = torch.cat([query_data, padding], dim=-1)

        # Flatten for autoencoder
        query_flat = query_data.view(-1, self.ae_input_dim)

        # Encode query data to code layer
        code_layer = self.autoencoder.encoder(query_flat)
        if torch.isnan(code_layer).any():
            print(f"WARNING: NaN in code_layer after encoder!")
        code_layer = code_layer.view(batch_size, -1, self.ae_code_dim)

        # Project embedded rows to code dimension for cross-aerial_with_attention
        keys_values = self.query_to_table_attention(embedded_rows)  # (batch_size, n_rows, ae_code_dim)
        if torch.isnan(keys_values).any():
            print(f"WARNING: NaN in keys_values after projection!")

        # Cross-aerial_with_attention: code layer (query) attends to embedded table rows (key, value)
        # NOTE: key_padding_mask in PyTorch is True for PADDING (opposite convention)
        # But we're passing True for VALID rows, so we need to INVERT the mask
        if table_mask is not None:
            padding_mask = ~table_mask  # Invert: True = padding, False = valid
        else:
            padding_mask = None

        attended_code, attention_weights = self.cross_attention(
            query=code_layer,
            key=keys_values,
            value=keys_values,
            key_padding_mask=padding_mask
        )
        if torch.isnan(attended_code).any():
            print(f"WARNING: NaN in attended_code after cross-aerial_with_attention!")

        # Flatten for decoder
        attended_flat = attended_code.view(-1, self.ae_code_dim)

        # Decode to reconstruct original query data
        reconstructed_flat = self.autoencoder.decoder(attended_flat)
        if torch.isnan(reconstructed_flat).any():
            print(f"WARNING: NaN in reconstructed_flat after decoder!")

        # Apply softmax per feature range (PyAerial style) on the flat tensor
        chunks = [reconstructed_flat[:, range_obj.start:range_obj.stop] for range_obj in feature_value_indices]
        softmax_chunks = [torch.softmax(chunk, dim=1) for chunk in chunks]
        reconstructed_flat = torch.cat(softmax_chunks, dim=1)

        # Reshape back to (batch_size, n_query, features)
        reconstructed = reconstructed_flat.view(batch_size, -1, reconstructed_flat.shape[1])

        # Truncate to original query dimension if needed
        if query_dim < reconstructed.shape[2]:
            reconstructed = reconstructed[:, :, :query_dim]

        return reconstructed, attended_code


def train_on_tables(model, table, query, feature_value_indices, epochs=100, lr=5e-3,
                    noise_factor=0.5, device='cpu'):
    """
    Train the model following PyAerial's training procedure.
    Foundation model handles variable table sizes through masking.

    Args:
        model: TabularFoundationModel instance
        table: Table data (n_rows, n_features)
        query: Query data to reconstruct (n_query, query_features)
        feature_value_indices: List of ranges for softmax application
        epochs: Number of training epochs
        lr: Learning rate (PyAerial default: 5e-3)
        noise_factor: Noise factor for denoising autoencoder (PyAerial default: 0.5)
        device: Device to train on

    Returns:
        trained_model: Trained model
        probability_matrix: Reconstruction probability matrix (n_query, query_features)
    """
    model = model.to(device)
    model.train()

    table_tensor = torch.FloatTensor(table).unsqueeze(0).to(device)  # (1, n_rows, n_features)
    n_rows = table.shape[0]

    # Create mask for valid rows (all rows are valid in single-table training)
    table_mask = torch.ones(1, n_rows, dtype=torch.bool, device=device)

    # Convert query data
    query_tensor = torch.FloatTensor(query).unsqueeze(0).to(device)  # (1, n_query, query_features)

    # PyAerial optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-8)
    criterion = nn.BCELoss()  # PyAerial uses BCELoss directly (output is already softmaxed)

    # Convert feature_value_indices to ranges
    softmax_ranges = [range(f['start'], f['end']) for f in feature_value_indices]

    for epoch in range(epochs):
        # Add noise to query (PyAerial's denoising approach)
        noisy_query = (query_tensor + torch.randn_like(query_tensor) * noise_factor).clamp(0, 1)

        # Forward pass with mask
        reconstructed, attended_code = model(table_tensor, noisy_query, softmax_ranges, table_mask=table_mask)

        # Debug: Check reconstruction range
        if epoch == 0:
            print(f"  - Reconstructed range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
            print(f"  - Reconstructed shape: {reconstructed.shape}, Query shape: {query_tensor.shape}")

        # Reconstruction loss
        loss = criterion(reconstructed, query_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    # Generate probability matrix (already softmaxed, no additional transformation needed)
    model.eval()
    with torch.no_grad():
        probability_matrix, _ = model(table_tensor, query_tensor, softmax_ranges, table_mask=table_mask)

    return model, probability_matrix


def generate_aerial_test_matrix(n_features, classes_per_feature, max_antecedents=2):
    """
    Generate test matrix (query) - creates ALL antecedent combinations at once.
    Unlike PyAerial which generates incrementally, we create all combinations upfront.

    Args:
        n_features: Number of features (columns) in the data
        classes_per_feature: List of number of classes for each feature
        max_antecedents: Maximum number of antecedents to combine

    Returns:
        test_matrix: numpy array of shape (n_test_vectors, total_dimensions)
        test_descriptions: List of tuples (feature_idx, class_idx) describing antecedents
        feature_value_indices: List of dicts with 'start', 'end', 'feature' for each feature
    """
    # Calculate total input dimension
    total_dim = sum(classes_per_feature)

    # Create feature_value_indices
    feature_value_indices = []
    start_idx = 0
    for feat_idx, n_classes in enumerate(classes_per_feature):
        feature_value_indices.append({
            'start': start_idx,
            'end': start_idx + n_classes,
            'feature': feat_idx
        })
        start_idx += n_classes

    # Initialize unmarked features with equal probabilities
    unmarked_features = _initialize_input_vectors(total_dim, feature_value_indices)

    test_vectors = []
    test_descriptions = []

    # Generate ALL combinations at once for all antecedent lengths
    # This is more efficient than PyAerial's incremental approach
    for r in range(1, max_antecedents + 1):
        # Get all feature combinations of size r
        for feature_indices in combinations(range(n_features), r):
            # For each feature combination, get all class combinations
            class_ranges = [list(range(classes_per_feature[f_idx])) for f_idx in feature_indices]

            # Generate all class combinations using product
            for class_combo in product(*class_ranges):
                # Create test vector
                test_vec = unmarked_features.copy()

                # Mark each selected feature-class pair
                description = []
                for feat_idx, class_idx in zip(feature_indices, class_combo):
                    feat_info = feature_value_indices[feat_idx]
                    # Set all classes of this feature to 0
                    test_vec[feat_info['start']:feat_info['end']] = 0.0
                    # Set the selected class to 1
                    test_vec[feat_info['start'] + class_idx] = 1.0
                    description.append((feat_idx, class_idx))

                test_vectors.append(test_vec)
                test_descriptions.append(tuple(description))

    return np.array(test_vectors), test_descriptions, feature_value_indices


def _initialize_input_vectors(input_vector_size, categories):
    """
    Initialize the input vectors with equal probabilities for each feature range.
    This follows PyAerial's initialization logic.
    """
    vector_with_unmarked_features = np.zeros(input_vector_size)
    for category in categories:
        vector_with_unmarked_features[category['start']:category['end']] = 1 / (
                category['end'] - category['start'])
    return vector_with_unmarked_features


# _mark_features function removed - no longer needed with simplified generation


def extract_rules_from_reconstruction(prob_matrix, test_descriptions, feature_value_indices,
                                      ant_similarity=0.5, cons_similarity=0.8, feature_names=None, debug=False):
    """
    Extract association rules from reconstruction probability matrix following PyAerial logic.

    Args:
        prob_matrix: Reconstruction probability matrix (n_test_vectors, total_dim)
        test_descriptions: List of tuples describing antecedents for each test vector
        feature_value_indices: List of dicts with 'start', 'end', 'feature' for each feature
        ant_similarity: Threshold for antecedent validation (default 0.5)
        cons_similarity: Threshold for consequent extraction (default 0.8)
        feature_names: Optional list of feature names for readable output
        debug: Print debug information

    Returns:
        association_rules: List of dicts with 'antecedents' and 'consequent'
    """
    association_rules = []
    low_support_count = 0
    no_consequent_count = 0

    for i, antecedent_desc in enumerate(test_descriptions):
        # Get reconstruction probabilities for this test vector
        implication_probabilities = prob_matrix[i]

        # Convert antecedent description to indices
        candidate_antecedents = []
        for feat_idx, class_idx in antecedent_desc:
            feat_info = feature_value_indices[feat_idx]
            ant_idx = feat_info['start'] + class_idx
            candidate_antecedents.append(ant_idx)

        # Check if antecedents have high support (PyAerial's validation)
        low_support = False
        for ant_idx in candidate_antecedents:
            if implication_probabilities[ant_idx] <= ant_similarity:
                low_support = True
                break

        if low_support:
            low_support_count += 1
            continue

        # Find high-support consequents (not in antecedents)
        consequent_list = []
        for feat_info in feature_value_indices:
            feat_probs = implication_probabilities[feat_info['start']:feat_info['end']]

            # Find class with highest probability for this feature
            max_class_idx = np.argmax(feat_probs)
            max_prob = feat_probs[max_class_idx]
            global_idx = feat_info['start'] + max_class_idx

            # Check if this is a valid consequent
            if max_prob >= cons_similarity and global_idx not in candidate_antecedents:
                consequent_list.append((feat_info['feature'], max_class_idx, max_prob))

        # Create rules for each consequent
        if consequent_list:
            antecedent_str = []
            for feat_idx, class_idx in antecedent_desc:
                if feature_names:
                    antecedent_str.append(f"{feature_names[feat_idx]}={class_idx}")
                else:
                    antecedent_str.append(f"F{feat_idx}={class_idx}")

            for cons_feat, cons_class, cons_prob in consequent_list:
                if feature_names:
                    consequent_str = f"{feature_names[cons_feat]}={cons_class}"
                else:
                    consequent_str = f"F{cons_feat}={cons_class}"

                association_rules.append({
                    'antecedents': antecedent_str,
                    'consequent': consequent_str,
                    'confidence': float(cons_prob)
                })
        else:
            no_consequent_count += 1

    if debug:
        print(f"  Debug: {low_support_count} test vectors had low antecedent support")
        print(f"  Debug: {no_consequent_count} test vectors had no valid consequents")

    return association_rules


def prepare_categorical_data(X):
    """
    One-hot encode categorical data and track feature information.

    Args:
        X: pandas DataFrame with categorical features

    Returns:
        encoded_data: numpy array of one-hot encoded data
        classes_per_feature: list of number of classes per feature
        feature_names: list of original feature names
    """
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(X)

    # Get number of classes per feature
    classes_per_feature = [len(cats) for cats in encoder.categories_]
    feature_names = X.columns.tolist()

    return encoded_data, classes_per_feature, feature_names, encoder


# Example usage
if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo

    # Hyperparameters for TabPFN-style Foundation Model with PyAerial autoencoder
    INPUT_DIM = 100  # Max input dimension for table columns
    EMBED_DIM = 25  # Embedding dimension for table rows
    MAX_ROWS = 1000  # Max number of rows supported (rulepfn model constraint)
    NUM_HEADS = 5
    NUM_LAYERS = 3
    MAX_ANTECEDENTS = 2

    # PyAerial training parameters
    NOISE_FACTOR = 0.5  # Noise for denoising autoencoder

    # PyAerial similarity thresholds (relaxed for initial testing)
    ANT_SIMILARITY = 0.2
    CONS_SIMILARITY = 0.8

    # Fetch datasets from UCI ML repository
    print("=" * 80)
    print("TabPFN-style Rule Learning with PyAerial Logic")
    print("=" * 80)
    print("\nFetching datasets from UCI ML repository...")

    # Breast Cancer dataset (ID: 14)
    breast_cancer = fetch_ucirepo(id=14)
    X_breast = breast_cancer.data.features

    # Congressional Voting Records dataset (ID: 105)
    voting = fetch_ucirepo(id=105)
    X_voting = voting.data.features

    # One-hot encode the datasets
    print("\nOne-hot encoding datasets...")
    breast_encoded, breast_classes, breast_features, breast_encoder = prepare_categorical_data(X_breast)
    voting_encoded, voting_classes, voting_features, voting_encoder = prepare_categorical_data(X_voting)

    print(f"\nBreast Cancer Dataset:")
    print(f"  - Encoded shape: {breast_encoded.shape}")
    print(f"  - Number of features: {len(breast_classes)}")
    print(f"  - Classes per feature: {breast_classes}")
    print(f"\nCongressional Voting Dataset:")
    print(f"  - Encoded shape: {voting_encoded.shape}")
    print(f"  - Number of features: {len(voting_classes)}")
    print(f"  - Classes per feature: {voting_classes}")

    # Limit ONE-HOT ENCODED columns to 100 (not the number of original features)
    MAX_ONEHOT_COLS = 100

    if breast_encoded.shape[1] > MAX_ONEHOT_COLS:
        print(f"\nLimiting Breast Cancer to {MAX_ONEHOT_COLS} one-hot encoded columns...")
        # Find how many features we can keep
        cumsum = np.cumsum(breast_classes)
        n_features_to_keep = np.searchsorted(cumsum, MAX_ONEHOT_COLS, side='right')
        n_dims = sum(breast_classes[:n_features_to_keep])
        breast_encoded = breast_encoded[:, :n_dims]
        breast_classes = breast_classes[:n_features_to_keep]
        breast_features = breast_features[:n_features_to_keep]
        print(f"  - Kept {n_features_to_keep} features with {n_dims} one-hot encoded columns")

    if voting_encoded.shape[1] > MAX_ONEHOT_COLS:
        print(f"\nLimiting Congressional Voting to {MAX_ONEHOT_COLS} one-hot encoded columns...")
        cumsum = np.cumsum(voting_classes)
        n_features_to_keep = np.searchsorted(cumsum, MAX_ONEHOT_COLS, side='right')
        n_dims = sum(voting_classes[:n_features_to_keep])
        voting_encoded = voting_encoded[:, :n_dims]
        voting_classes = voting_classes[:n_features_to_keep]
        voting_features = voting_features[:n_features_to_keep]
        print(f"  - Kept {n_features_to_keep} features with {n_dims} one-hot encoded columns")

    # Prepare both datasets for training
    datasets = [
        ("Breast Cancer", breast_encoded, breast_classes, breast_features),
        ("Congressional Voting", voting_encoded, voting_classes, voting_features)
    ]

    # Step 1: Train on both tables
    print("\n" + "=" * 80)
    print("STEP 1: Training on Both Datasets")
    print("=" * 80)

    trained_models = []
    test_matrices = []
    test_descriptions_list = []
    feature_value_indices_list = []

    for dataset_name, encoded_data, classes_per_feat, feature_names in datasets:
        print("\n" + "=" * 80)
        print(f"Processing {dataset_name} Dataset")
        print("=" * 80)

        # Foundation model handles variable row counts up to MAX_ROWS
        # No need to manually limit rows - the model uses masking
        print(f"\nDataset info:")
        print(f"  - Shape: {encoded_data.shape}")
        print(f"  - Max rows supported by model: {MAX_ROWS}")

        if encoded_data.shape[0] > MAX_ROWS:
            print(f"  - WARNING: Dataset has {encoded_data.shape[0]} rows, limiting to {MAX_ROWS}")
            encoded_data = encoded_data[:MAX_ROWS]

        # Generate test matrix (query) following PyAerial logic
        print(f"\nGenerating PyAerial test matrix with max {MAX_ANTECEDENTS} antecedents...")
        test_matrix, test_descriptions, feature_value_indices = generate_aerial_test_matrix(
            n_features=len(classes_per_feat),
            classes_per_feature=classes_per_feat,
            max_antecedents=MAX_ANTECEDENTS
        )

        print(f"  - Test matrix shape: {test_matrix.shape}")
        print(f"  - Number of test vectors: {len(test_descriptions)}")

        # Initialize FOUNDATION model with max_rows constraint
        model = TabularFoundationModel(
            input_dim=INPUT_DIM,
            embed_dim=EMBED_DIM,
            ae_input_dim=INPUT_DIM,
            ae_feature_count=len(classes_per_feat),
            max_rows=MAX_ROWS,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS
        )

        print(f"\nTraining TabPFN-style model on {dataset_name}...")
        print(f"  - AutoEncoder dimensions: {model.autoencoder.dimensions}")
        trained_model, _ = train_on_tables(
            model, encoded_data, test_matrix, feature_value_indices,
            epochs=100, lr=5e-3, noise_factor=NOISE_FACTOR, device='cpu'
        )

        # Store for later rule extraction
        trained_models.append(trained_model)
        test_matrices.append(test_matrix)
        test_descriptions_list.append(test_descriptions)
        feature_value_indices_list.append(feature_value_indices)

        print(f"  - Training completed!")
        print(f"\n{'-' * 80}")

    # Step 2: Extract rules from trained models
    print("\n" + "=" * 80)
    print("STEP 2: Extracting Rules from Trained Models")
    print("=" * 80)

    for idx, (dataset_info, trained_model, test_matrix, test_descriptions, feature_value_indices) in enumerate(
            zip(datasets, trained_models, test_matrices, test_descriptions_list, feature_value_indices_list)):

        dataset_name, encoded_data, classes_per_feat, feature_names = dataset_info

        print("\n" + "=" * 80)
        print(f"Extracting Rules for {dataset_name} Dataset")
        print("=" * 80)

        # Ensure data doesn't exceed MAX_ROWS
        if encoded_data.shape[0] > MAX_ROWS:
            encoded_data = encoded_data[:MAX_ROWS]

        # Forward pass to get probability matrix
        print(f"\nRunning forward pass to get probability matrix...")
        trained_model.eval()
        with torch.no_grad():
            table_tensor = torch.FloatTensor(encoded_data).unsqueeze(0).to('cpu')
            query_tensor = torch.FloatTensor(test_matrix).unsqueeze(0).to('cpu')
            softmax_ranges = [range(f['start'], f['end']) for f in feature_value_indices]

            # Create mask for all valid rows
            n_rows = encoded_data.shape[0]
            table_mask = torch.ones(1, n_rows, dtype=torch.bool, device='cpu')

            prob_matrix, _ = trained_model(table_tensor, query_tensor, softmax_ranges, table_mask=table_mask)
            prob_matrix = prob_matrix.squeeze(0).detach().cpu().numpy()

        print(f"  - Reconstruction probability matrix shape: {prob_matrix.shape}")
        print(f"  - Probability range: [{prob_matrix.min():.4f}, {prob_matrix.max():.4f}]")
        print(f"  - Probability mean: {prob_matrix.mean():.4f}")

        # Extract rules using PyAerial logic
        print(f"\nExtracting association rules (ant_sim={ANT_SIMILARITY}, cons_sim={CONS_SIMILARITY})...")
        association_rules = extract_rules_from_reconstruction(
            prob_matrix=prob_matrix,
            test_descriptions=test_descriptions,
            feature_value_indices=feature_value_indices,
            ant_similarity=ANT_SIMILARITY,
            cons_similarity=CONS_SIMILARITY,
            feature_names=feature_names,
            debug=True
        )

        print(f"  - Total rules extracted: {len(association_rules)}")

        # Display first 10 rules
        print(f"\nFirst 10 Association Rules:")
        for rule_idx, rule in enumerate(association_rules[:10], 1):
            antecedents_str = " AND ".join(rule['antecedents'])
            print(f"  Rule {rule_idx}: {antecedents_str} => {rule['consequent']} "
                  f"(confidence: {rule['confidence']:.3f})")

        print(f"\n{'-' * 80}")

    # Save last model for future training
    # if trained_models:
    #     torch.save(trained_models[-1].state_dict(), "tabular_foundation_model.pt")
    #     print("\n" + "=" * 80)
    #     print("Model saved as 'tabular_foundation_model.pt'")
    #     print("=" * 80)
