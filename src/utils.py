import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import config

def get_device() -> torch.device:
    """
    Returns the computing device configured in config.py.

    Returns:
        torch.device: The device (CPU or CUDA) where tensors will be allocated.
    """
    return torch.device(config.DEVICE)

def load_labels(path: str = config.LABELS_PATH) -> pd.DataFrame:
    """
    Loads and normalizes the ground-truth personality labels from a pickle file.

    This function handles the specific nested dictionary format often used in 
    ChaLearn datasets and converts it into a clean Pandas DataFrame.

    Args:
        path (str): The absolute path to the .pkl annotation file. 
                    Defaults to config.LABELS_PATH.

    Returns:
        pd.DataFrame: A DataFrame containing the metadata and scores.
            Columns typically include:
            ['video_name', 'openness', 'conscientiousness', 'extraversion', 
             'agreeableness', 'neuroticism', 'interview']

    Raises:
        FileNotFoundError: If the pickle file does not exist at the specified path.
        ValueError: If the unpacked data is not a dictionary as expected.
    """
    print(f"Loading labels from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found at {path}. Check config.py!")

    with open(path, 'rb') as f:
        # 'latin1' encoding is required for older Python 2 pickles often found in datasets
        data = pickle.load(f, encoding='latin1')

        # Data is a dictionary. Let's inspect the keys to decide how to load it.
    keys = list(data.keys())
    first_key = str(keys[0])
    
    print(f"  Detected pickle keys (first 3): {keys[:3]}")

    # SCENARIO A: Trait-First Structure (e.g. {'openness': {'vid1': 0.5}, ...})
    # This is likely what you have.
    if 'openness' in keys or 'extraversion' in keys:
        print("  Detected Trait-First structure. Pivoting...")
        # pd.DataFrame(data) automatically aligns keys as Columns and nested keys as Index
        df = pd.DataFrame(data)
        df.index.name = 'video_name'
        df.reset_index(inplace=True)
        return df

    # SCENARIO B: Video-First Structure (e.g. {'vid1.mp4': {'openness': 0.5}, ...})
    elif first_key.endswith('.mp4'):
        print("  Detected Video-First structure.")
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'video_name'
        df.reset_index(inplace=True)
        return df

    # SCENARIO C: Unknown
    else:
        # Fallback: Try standard DataFrame constructor and hope for the best
        print("   ⚠️ Unknown structure. Attempting standard load.")
        try:
            df = pd.DataFrame(data)
            if 'video_name' not in df.columns:
                df.index.name = 'video_name'
                df.reset_index(inplace=True)
            return df
        except:
            raise ValueError(f"Label file format not recognized. Keys look like: {keys[:5]}")

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filename: str = "best_model.pth") -> None:
    """
    Saves the model weights and optimizer state to disk.

    Args:
        model (nn.Module): The PyTorch model instance to save.
        optimizer (optim.Optimizer): The optimizer (captures learning rate/momentum state).
        filename (str): The name of the file to save within config.CHECKPOINT_DIR.
    """
    save_path = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")