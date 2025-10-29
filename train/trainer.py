import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import json

def train_siglip(model, train_dataset, val_dataset=None, 
                 batch_size=2,  # Reduced batch size
                 num_epochs=10, 
                 learning_rate=1e-4,
                 gradient_accumulation_steps=4):  # Increased accumulation steps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=1)  # Reduced workers
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (images, texts) in enumerate(tqdm(train_loader)):
            # Clear cache at the start of each batch
            torch.cuda.empty_cache()
            
            images = images.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                logits = model(images, texts)
                labels = torch.arange(len(images)).to(device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Clear cache after optimizer step
                torch.cuda.empty_cache()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}") 
    
    # After training completes, save everything we need for VLM training
    print("\nSaving model and processed dataset for VLM training...")
    
    # Create directories
    save_base_dir = "siglip_processed_data"
    os.makedirs(save_base_dir, exist_ok=True)
    
    # 1. Save the trained SigLIP model
    model_path = f"{save_base_dir}/siglip_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved SigLIP model to {model_path}")
    
    # 2. Process and save dataset with embeddings
    model.eval()
    processed_dataset = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(train_dataset)), desc="Processing dataset"):
            image, text = train_dataset[idx]
            
            # Get image path and label from the dataset
            image_path = train_dataset.data[idx]['image']
            label = image_path.split('_')[2]  # Extract label from path
            
            # Generate embeddings
            image = image.unsqueeze(0).to(device)
            image_embedding = model.encode_image(image)
            text_embedding = model.encode_text([text], device)
            
            # Create sample dictionary
            sample = {
                'idx': idx,
                'label': label,
                'text': text,
                'image_path': image_path,
                'embeddings': {
                    'image': image_embedding.cpu().numpy().tolist(),
                    'text': text_embedding.cpu().numpy().tolist()
                }
            }
            
            # Save individual sample embeddings
            np.savez(
                f"{save_base_dir}/sample_{idx}_embeddings.npz",
                image_embedding=image_embedding.cpu().numpy(),
                text_embedding=text_embedding.cpu().numpy()
            )
            
            processed_dataset.append(sample)
    
    # 3. Save dataset metadata
    metadata = {
        'num_samples': len(processed_dataset),
        'embedding_dim': image_embedding.shape[-1],
        'model_path': model_path
    }
    
    # Save metadata
    with open(f"{save_base_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save processed dataset
    with open(f"{save_base_dir}/processed_dataset.json", 'w') as f:
        json.dump(processed_dataset, f, indent=2)
    
    print(f"\nSaved processed dataset and embeddings to {save_base_dir}/")
    print("Directory structure:")
    print(f"  {save_base_dir}/")
    print(f"  ├── siglip_model.pth")
    print(f"  ├── metadata.json")
    print(f"  ├── processed_dataset.json")
    print(f"  └── sample_*_embeddings.npz")
    
    return model, processed_dataset 