import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
model_name = "TinyPixel/Llama-2-7B-bf16-sharded"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "TinyPixel/Llama-2-7B-bf16-sharded"



def main():
    parser = argparse.ArgumentParser(description="Text generation using a GPT-like model.")
    parser.add_argument("text", type=str, help="Input text for generation.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for model inference. Defaults to 'cuda:0'.")
    
    args = parser.parse_args()
    text = args.text
    device = args.device

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False

    lora_config = LoraConfig.from_pretrained('outputs')
    model = get_peft_model(model, lora_config)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Human: ", text)
    print("Assistant: ", generated_text)

if __name__ == "__main__":
    main()
