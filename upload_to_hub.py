from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

def upload_to_hub(
    model_path="vlm_model",
    repo_name="jatingocodeo/phi-vlm",
    token=None
):
    print("Loading base model...")
    # Load the base model first with CPU offload
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        device_map="cpu",  # Load to CPU first
        torch_dtype=torch.float32
    )
    
    print("Loading LoRA adapter...")
    # Load the LoRA adapter
    peft_config = PeftConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(
        base_model, 
        model_path,
        device_map="cpu",  # Load to CPU
        torch_dtype=torch.float32
    )
    
    print("Loading tokenizer...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Creating repository...")
    # Create the repo
    api = HfApi()
    try:
        create_repo(repo_name, private=False, token=token)
    except Exception as e:
        print(f"Repo might already exist: {e}")
    
    print("Pushing model to hub...")
    # Save model and tokenizer to hub
    model.push_to_hub(
        repo_name, 
        token=token,
        max_shard_size="500MB",
        safe_serialization=True
    )
    
    print("Pushing tokenizer to hub...")
    tokenizer.push_to_hub(repo_name, token=token)
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import os
    # Get token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Please set the HF_TOKEN environment variable with your Hugging Face token")
        exit(1)
    
    upload_to_hub(token=hf_token) 