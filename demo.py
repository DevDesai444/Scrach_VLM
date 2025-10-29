from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import requests
from io import BytesIO

def load_model():
    """Load the SmolVLM2 model and processor."""
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False  # Disable Flash Attention 2
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

def process_image(image_url, question, model, processor):
    """Process an image and generate a response to a question."""
    # Download and save the image
    response = requests.get(image_url)
    image_name = "sample_image.jpg"
    with open(image_name, "wb") as f:
        f.write(response.content)
    print(f"Image saved as {image_name}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def process_video(video_path, question, model, processor):
    """Process a video and generate a response to a question."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": question}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main():
    # Load model
    print("Loading model...")
    model, processor = load_model()
    
    # Example image processing
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    question = "Can you describe this image?"
    
    print("\nProcessing image...")
    response = process_image(image_url, question, model, processor)
    print(f"Question: {question}")
    print(f"Response: {response}")
    
    # Example video processing (uncomment and provide video path to test)
    """
    video_path = "path_to_your_video.mp4"
    video_question = "Describe this video in detail"
    
    print("\nProcessing video...")
    video_response = process_video(video_path, video_question, model, processor)
    print(f"Question: {video_question}")
    print(f"Response: {video_response}")
    """

if __name__ == "__main__":
    main() 