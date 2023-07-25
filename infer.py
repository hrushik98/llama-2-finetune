import argparse
import torch
from transformers import LoraConfig

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from peft import LoraConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser(description="Text generation using a GPT-like model.")
    parser.add_argument("text", type=str, help="Input text for generation.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for model inference. Defaults to 'cuda:0'.")
    
    args = parser.parse_args()
    text = args.text
    device = args.device

    lora_config = LoraConfig.from_pretrained('outputs')
    model = get_peft_model(model, lora_config)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Input Text: ", text)
    print("Generated Text: ", generated_text)

if __name__ == "__main__":
    main()
