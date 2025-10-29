import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from PIL import Image
import torchvision.datasets as datasets
import os

def load_model(model_id):
    # First load the base model
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model for CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",  # Force CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Enable memory optimization
    )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(
        base_model, 
        model_id,
        device_map="cpu"  # Force CPU
    )
    
    return model, tokenizer

def generate_description(image, model, tokenizer, max_length=100, temperature=0.7, top_p=0.9):
    try:
        # Convert and resize image
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((32, 32))
        
        # Format the input text
        input_text = """Below is an image. Please describe it in detail.

Image: [IMAGE]
Description: """
        
        # Ensure we have valid token IDs
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Generate response with simpler parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=False,  # Disable caching to avoid the error
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.split("Description: ")[-1].strip()
    
    except Exception as e:
        import traceback
        return f"Error generating description: {str(e)}\n{traceback.format_exc()}"

def create_demo(model_id):
    # Load model and tokenizer
    model, tokenizer = load_model(model_id)
    
    # Get CIFAR10 examples
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True)
    examples = []
    used_classes = set()
    
    for idx in range(len(cifar10_test)):
        img, label = cifar10_test[idx]
        class_name = cifar10_test.classes[label]
        if class_name not in used_classes:
            examples.append(img)
            used_classes.add(class_name)
        if len(used_classes) == 10:
            break
    
    # Define the interface function
    def process_image(image, max_length, temperature, top_p):
        try:
            return generate_description(
                image,
                model,
                tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            return f"Error generating description: {str(e)}"
    
    # Create the interface
    demo = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.Slider(
                minimum=50,
                maximum=200,
                value=100,
                step=10,
                label="Maximum Length"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                label="Top P"
            )
        ],
        outputs=gr.Textbox(label="Generated Description", lines=5),
        title="Image Description Generator",
        description="""This model generates detailed descriptions of images.
        
        You can adjust the generation parameters:
        - **Maximum Length**: Controls the length of the generated description
        - **Temperature**: Higher values make the description more creative
        - **Top P**: Controls the randomness in word selection
        """,
        examples=[[ex] for ex in examples]
    )
    return demo

if __name__ == "__main__":
    # Use your model ID
    model_id = "jatingocodeo/phi-vlm"
    demo = create_demo(model_id)
    demo.launch() 