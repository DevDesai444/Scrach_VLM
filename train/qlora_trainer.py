from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import Dataset
from tqdm import tqdm
from train.vlm_dataset import VLMDataset
from model.phi_with_vision import PhiWithVision

def prepare_qlora_training(siglip_data_dir, output_dir="vlm_model"):
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32
    )

    # Load model with quantization config and attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager"  # Add this to handle flash-attention warning
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",  # Match the model
        trust_remote_code=True
    )
    
    # Wrap the model with PhiWithVision
    model = PhiWithVision(model)
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_vlm(siglip_data_dir, output_dir="vlm_model", batch_size=4, num_epochs=3):
    # Now VLMDataset will be recognized
    dataset = VLMDataset(siglip_data_dir)
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_qlora_training(siglip_data_dir, output_dir)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch"
    )
    
    # Custom training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(range(len(dataset) // batch_size))
        
        for i in range(0, len(dataset), batch_size):
            batch_samples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            
            # Prepare inputs
            input_texts = [sample['input_text'] for sample in batch_samples]
            target_texts = [sample['target_text'] for sample in batch_samples]
            image_embeddings = torch.stack([sample['image_embeddings'] for sample in batch_samples])
            
            # Tokenize with truncation
            inputs = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=128,  # Match dataset max_length
                return_tensors="pt"
            )
            
            targets = tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=128,  # Match dataset max_length
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = targets.input_ids.to(device)
            image_embeddings = image_embeddings.to(device)
            
            # Forward pass
            outputs = model(
                **inputs,
                labels=labels,
                image_embeddings=image_embeddings
            )
            
            loss = outputs.loss
            loss.backward()
            
            if (i // batch_size + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.update(1)
            
        # Save checkpoint
        model.save_pretrained(f"{output_dir}/checkpoint-epoch-{epoch}")
        
        avg_loss = total_loss / (len(dataset) // batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Saving model...")
    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer 