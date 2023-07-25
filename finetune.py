import argparse
import csv
from datasets import Dataset, Features, Value
import gdown
import pandas as pd

def download_google_sheet(google_sheet_url, output_file):
    # Extract the Google Sheet ID from the URL
    google_sheet_id = google_sheet_url.split('/')[-2]

    # Download the Google Sheet as a CSV file
    url = f'https://drive.google.com/uc?id={google_sheet_id}'
    gdown.download(url, output_file, quiet=False)

if __name__ == "__main__":
    # Set up argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and process Google Sheet as a CSV dataset.")
    parser.add_argument("google_sheet_url", type=str, help="Google Drive link to the CSV file.")
    args = parser.parse_args()

    # Download the Google Sheet as a CSV file
    output_file = 'raw_data.csv'  # The name of the file to save the downloaded data
    download_google_sheet(args.google_sheet_url, output_file)

    # Process the CSV file as a dataset
    data = pd.read_csv(output_file)

    data['text'] = "-"
    for i in range(0, len(data)):
        data['Human'][i] = "### Human: " + data['Human'][i]
        data['Assistant'][i] = "### Assistant: " + data['Assistant'][i]
        data['text'][i] = data['Human'][i] + data['Assistant'][i]

    data = data['text']
    dataset = Dataset.from_pandas(data, features=Features({
        "Human": Value("string"),
        "Assistant": Value("string"),
        "text": Value("string")
    }))
    dataset_rows = dataset['text']
    dataset = Dataset.from_dict({"text": dataset_rows})




from datasets import load_dataset


import pandas as pd
from datasets import Dataset, Features, Value

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

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


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)


from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")
