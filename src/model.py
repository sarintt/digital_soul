import torch
import torch.nn as nn
import config

class DigitalSoulModel(nn.Module):
    """
    A Multi-Modal Neural Network for Personality Prediction.

    Architecture:
    1. Acoustic Branch: An LSTM that processes temporal MFCC features.
    2. Linguistic Branch: Direct input of pre-computed Transformer embeddings.
    3. Fusion Layer: Concatenates the two streams.
    4. Prediction Head: A Multi-Layer Perceptron (MLP) mapping fused features to 5 traits.

    Args:
        None: Uses hyperparameters defined in config.py.
    """

    def __init__(self):
        super(DigitalSoulModel, self).__init__()
        
        # --- Acoustic Branch (LSTM) ---
        # Input Shape: (Batch_Size, Sequence_Length, 13)
        self.acoustic_lstm = nn.LSTM(
            input_size=13, 
            hidden_size=config.HIDDEN_DIM, 
            batch_first=True,
            num_layers=2,
            dropout=config.DROPOUT
        )
        
        # --- Fusion Layer ---
        # RoBERTa (768 dim) + LSTM Last Hidden State (config.HIDDEN_DIM)
        fusion_dim = 768 + config.HIDDEN_DIM
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5) # Output: 5 Personality Traits
        )
        
        # Sigmoid activation forces the regression output to be between 0.0 and 1.0
        self.sigmoid = nn.Sigmoid()

    def forward(self, ling_input: torch.Tensor, acou_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            ling_input (Tensor): Shape (Batch, 768). The text embeddings.
            acou_input (Tensor): Shape (Batch, Time, 13). The audio features.

        Returns:
            Tensor: Shape (Batch, 5). The predicted personality scores.
        """
        # 1. Process Acoustic Stream
        # LSTM output is (out, (h_n, c_n)). 
        # We need h_n (hidden state) which summarizes the temporal sequence.
        _, (h_n, _) = self.acoustic_lstm(acou_input)
        
        # h_n shape is (Num_Layers, Batch, Hidden_Dim). 
        # We take the last layer's hidden state: index [-1]
        acou_vec = h_n[-1] 
        
        # 2. Feature Fusion
        # Concatenate along dimension 1 (feature dimension)
        combined = torch.cat((ling_input, acou_vec), dim=1)
        
        # 3. Prediction Head
        logits = self.fusion_head(combined)
        
        return self.sigmoid(logits)