import os
import torch
import numpy as np
from torch.utils.data import Dataset
import config
from src.utils import load_labels

class DigitalSoulDataset(Dataset):
    """
    A custom PyTorch Dataset for loading multi-modal personality data.

    This dataset manages the alignment between:
    1. Saved .npy Linguistic feature files (from ModernBERT)
    2. Saved .npy Acoustic feature files (from Librosa)
    3. Ground truth personality scores (from the labels CSV/Pickle)

    Attributes:
        valid_samples (list): A list of tuples containing (ling_path, acou_path, score_array).
                              Only samples where both feature files exist are included.
    """

    def __init__(self):
        """
        Initializes the dataset by loading labels and verifying feature file existence.
        """
        self.features_dir = config.FEATURES_DIR
        
        # 1. Load labels
        self.labels_df = load_labels()
        
        # 2. Filter: Only keep videos where we successfully extracted features
        self.valid_samples = []
        missing_ling = 0
        missing_acou = 0

        for idx, row in self.labels_df.iterrows():
            # The label DF typically has "video_name.mp4". 
            # We need to find the matching feature files.
            raw_filename = os.path.basename(row['video_name'])
            base_name = raw_filename.replace(".mp4", "")
            
            # FIXED: Naming convention now matches src/features.py 
            # (e.g., "video1_ling.npy" instead of "video1.wav_ling.npy")
            ling_file = os.path.join(self.features_dir, f"{base_name}_ling.npy")
            acou_file = os.path.join(self.features_dir, f"{base_name}_acou.npy")

            # Check existence
            has_ling = os.path.exists(ling_file)
            has_acou = os.path.exists(acou_file)

            if has_ling and has_acou:
                scores = row[['openness', 'conscientiousness', 'extraversion', 
                              'agreeableness', 'neuroticism']].values.astype(float)
                self.valid_samples.append((ling_file, acou_file, scores))
            else:
                if not has_ling: missing_ling += 1
                if not has_acou: missing_acou += 1
        
        print(f"Dataset Initialized.")
        print(f"  Valid Samples: {len(self.valid_samples)}")
        print(f"  Missing Linguistic: {missing_ling}")
        print(f"  Missing Acoustic: {missing_acou}")

        if len(self.valid_samples) == 0:
            raise ValueError(
                "No valid samples found! \n"
                "1. Check that you ran 'uv run main.py extract'.\n"
                "2. Check that your feature filenames in 'data/features' match the labels.\n"
                f"   (Example Label: {base_name}.mp4 -> Looking for: {base_name}_ling.npy)"
            )
        
        print(f"📊 Dataset Initialized. Found {len(self.valid_samples)} valid samples.")

    def __len__(self) -> int:
        """Returns the total number of valid samples."""
        return len(self.valid_samples)

    def __getitem__(self, idx: int):
        """
        Retrieves the multi-modal features and label for a specific index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (ling_t, acou_t, label_t)
                - ling_t (Tensor): Shape (768,). The RoBERTa embedding vector.
                - acou_t (Tensor): Shape (TimeFrames, 13). The MFCC sequence transposed for LSTM.
                - label_t (Tensor): Shape (5,). The Big Five personality scores (0.0 to 1.0).
        """
        ling_path, acou_path, scores = self.valid_samples[idx]
        
        # Load NPY files
        ling_vec = np.load(ling_path) # Shape (768,)
        acou_mat = np.load(acou_path) # Shape (13, Time)
        
        # Convert to Tensor
        ling_t = torch.tensor(ling_vec, dtype=torch.float32)
        acou_t = torch.tensor(acou_mat, dtype=torch.float32)
        
        # Transpose Acoustic for LSTM: 
        # Librosa gives (Features, Time), but PyTorch LSTM expects (Time, Features) if batch_first=True
        acou_t = acou_t.transpose(0, 1) 
        
        label_t = torch.tensor(scores, dtype=torch.float32)
        
        return ling_t, acou_t, label_t