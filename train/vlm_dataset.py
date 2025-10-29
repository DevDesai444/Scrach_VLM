import torch
from torch.utils.data import Dataset
import json
import numpy as np

class VLMDataset(Dataset):
    def __init__(self, siglip_data_dir):
        self.siglip_data_dir = siglip_data_dir  # Store the directory path
        
        # Load metadata and processed dataset
        with open(f"{siglip_data_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        with open(f"{siglip_data_dir}/processed_dataset.json", 'r') as f:
            self.data = json.load(f)
            
        # Template for instruction format
        self.instruction_template = """Below is an image. Please describe it in detail.

Image: <image>
Description: """
        
        # Maximum sequence length for input
        self.max_length = 128  # Adjust this based on model's requirements
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load embeddings using the stored directory path
        embeddings = np.load(f"{self.siglip_data_dir}/sample_{idx}_embeddings.npz")
        
        # Format input with instruction
        input_text = self.instruction_template
        target_text = sample['text']
        
        # Tokenize input and target (we'll do this in the training loop)
        return {
            'image_embeddings': torch.tensor(embeddings['image_embedding']),
            'input_text': input_text,
            'target_text': target_text,
            'label': sample['label']
        }

    def __len__(self):
        return len(self.data) 