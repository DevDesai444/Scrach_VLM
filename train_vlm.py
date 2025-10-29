from train.qlora_trainer import train_vlm

def main():
    # Path to saved SigLIP data
    siglip_data_dir = "siglip_processed_data"
    
    # Train VLM
    model = train_vlm(
        siglip_data_dir=siglip_data_dir,
        output_dir="vlm_model",
        batch_size=4,
        num_epochs=3
    )
    
    print("VLM training completed!")

if __name__ == "__main__":
    main() 