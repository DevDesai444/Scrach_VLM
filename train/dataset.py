import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import json
import os

class SigLIPDataset(Dataset):
    def __init__(self, json_path, transform=None):
        # Load our JSON descriptions
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Load CIFAR10 test dataset
        self.cifar_dataset = datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True
        )
        
        self.transform = transform
        print(f"SigLIP Dataset loaded with {len(self.data)} image-text pairs")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Debug prints for path parsing
        image_path = item['image']
        print(f"\nProcessing image path: {image_path}")
        
        parts = image_path.split('/')[-1].split('_')
        print(f"Split parts: {parts}")
        
        try:
            label = int(parts[2])
            print(f"Extracted label: {label}")
        except Exception as e:
            print(f"Error parsing label from parts {parts}")
            raise e
        
        # Find corresponding CIFAR10 image
        cifar_images_with_label = [(i, img) for i, (img, l) in enumerate(self.cifar_dataset) if l == label]
        if cifar_images_with_label:
            _, image = cifar_images_with_label[0]  # Take the first image with matching label
        else:
            raise ValueError(f"No CIFAR10 image found with label {label}")
        
        # Apply transforms directly (no need to convert to PIL Image as it's already in correct format)
        if self.transform:
            image = self.transform(image)
        
        # Get all assistant responses and combine them
        descriptions = []
        for conv in item['conversations']:
            if conv['from'] == 'assistant':
                descriptions.append(conv['value'])
        description = " ".join(descriptions)
        
        return image, description 