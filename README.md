# llama-2-finetune: Finetune Llama 2 7b Model

## Step-by-step Guide

1. **Prepare Your Dataset**

   Make sure your dataset has ONLY two columns: 'Human' and 'Assistant'.

2. **Upload the Dataset to Google Drive**

   - Upload the dataset to your Google Drive.
   - Click on the "Share" button for the uploaded file and change the access to "edit" for "Anyone with the link."
   - Copy the link to the file.

   ![Screen Shot 2023-07-25 at 10 01 31 AM](https://github.com/hrushik98/llama-2-finetune/assets/91076764/5bf9b201-f086-430f-8573-ded39d7ca70d)

3. **Clone the Repository**

   Run the following command to clone the llama-2-finetune repository:
   ```python
   !git clone https://github.com/hrushik98/llama-2-finetune.git 
   ```

4. **Install Dependencies**

Change the current directory to llama-2-finetune and install the required dependencies with the following commands:

  ```python
      %cd llama-2-finetune/
      !pip install -r requirements.txt    
  ```
5. **Start Finetuning**

Run the finetune.py script with your dataset link as an argument:
  ```python
  !python3 finetune.py "<dataset link>"  
  ```

## Infererence of the finetuned model
1. **After the finetuning is finished, the weights will be saved in the folder "outputs"**
2. ```python
   lora_config = LoraConfig.from_pretrained('outputs')
    model = get_peft_model(model, lora_config)

   text = "<enter the instruction>"
    device = "cuda:0"
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

