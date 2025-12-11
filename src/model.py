import torch
import torch.nn as nn
import config

class DigitalSoulModel(nn.Module):

    def __init__(self):
        super(DigitalSoulModel, self).__init__()

        # ---------- Acoustic branch ----------
        # Input acoustic feature dim from feature.py (MFCC13 + Δ + Δ² + RMS = 40)
        self.acou_input_dim = 40

        # Project 40-dim → H (ช่วยให้ LSTM เรียนง่ายขึ้น)
        self.acoustic_norm = nn.LayerNorm(self.acou_input_dim)
        self.acoustic_proj = nn.Linear(self.acou_input_dim, config.HIDDEN_DIM)
        self.acoustic_proj_act = nn.ReLU()

        # BiLSTM over projected features
        # LSTM input: (B, T, HIDDEN_DIM)
        self.acoustic_lstm = nn.LSTM(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
            dropout=config.DROPOUT,
        )

        # BiLSTM → hidden dim = 2 * HIDDEN_DIM
        self.acou_hidden_dim = 2 * config.HIDDEN_DIM

        # Dropout on LSTM outputs
        self.acoustic_dropout = nn.Dropout(config.DROPOUT)

        # Attention over time
        self.attn_proj = nn.Linear(self.acou_hidden_dim, self.acou_hidden_dim)
        self.attn_score = nn.Linear(self.acou_hidden_dim, 1)

        # ---------- Text branch ----------
        self.ling_dim = 768
        self.ling_norm = nn.LayerNorm(self.ling_dim)

        # ---------- Fusion (Gated) ----------
        # เราจะ project ทั้ง text & audio มาที่ latent space เดียวกัน
        self.fusion_dim = 256

        self.ling_proj = nn.Linear(self.ling_dim, self.fusion_dim)
        self.acou_proj = nn.Linear(self.acou_hidden_dim, self.fusion_dim)

        # Gating network: concat(text_proj, acou_proj) -> gate vector in [0,1]^fusion_dim
        self.gate_layer = nn.Linear(2 * self.fusion_dim, self.fusion_dim)

        # ---------- Prediction head ----------
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 5)  # raw logits (ยังไม่ sigmoid)
        )

    def forward(self, ling_input: torch.Tensor, acou_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ling_input: (B, 768)       - text embeddings
            acou_input: (B, T, 40)     - enriched acoustic features from feature.py

        Returns:
            logits: (B, 5)             - raw scores for 5 traits (0–1 หลัง sigmoid ภายนอก)
        """

        # ----- Text branch -----
        ling = self.ling_norm(ling_input)           # (B, 768)
        ling = self.ling_proj(ling)                 # (B, fusion_dim)

        # ----- Acoustic branch -----
        # Normalize per frame
        acou = self.acoustic_norm(acou_input)       # (B, T, 40)

        # Project 40 -> HIDDEN_DIM
        acou = self.acoustic_proj(acou)             # (B, T, HIDDEN_DIM)
        acou = self.acoustic_proj_act(acou)         # (B, T, HIDDEN_DIM)

        # BiLSTM
        # out: (B, T, 2*HIDDEN_DIM)
        out, _ = self.acoustic_lstm(acou)
        out = self.acoustic_dropout(out)

        # Attention over time
        proj = torch.tanh(self.attn_proj(out))      # (B, T, 2H)
        scores = self.attn_score(proj).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(scores, dim=1) # (B, T)

        attn_weights = attn_weights.unsqueeze(-1)   # (B, T, 1)
        acou_vec = torch.sum(out * attn_weights, dim=1)  # (B, 2H) = (B, acou_hidden_dim)

        # Project audio vector to fusion_dim
        acou_f = self.acou_proj(acou_vec)           # (B, fusion_dim)

        # ----- Gated Fusion -----
        # concat text & audio in fusion space
        fusion_cat = torch.cat([ling, acou_f], dim=1)          # (B, 2*fusion_dim)

        # gate vector per feature
        gate = torch.sigmoid(self.gate_layer(fusion_cat))      # (B, fusion_dim)

        # gated combination: gate * text + (1-gate) * audio
        fused = gate * ling + (1.0 - gate) * acou_f            # (B, fusion_dim)

        # ----- Prediction head -----
        logits = self.fusion_head(fused)                       # (B, 5)
        return logits
