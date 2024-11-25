from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import os
from huggingface_hub import login

login(token = os.getenv('HF_TOKEN')) # set HF_TOKEN in the environment for hugging face token to download required model 
model_name = "gpt2" # update model name to use different models from hugging face
device = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available

# load the dataset and curate input prompts and output responses to train the model
dataset = load_dataset('json', data_files='dataset.json')
input_prompts = []
output_responses = []
for item in dataset["train"]:
    input_prompt = f"<task> Generate a lesson for {item['type']} at difficulty level {item['difficulty']}.\n"
    output_response = f"<lesson> Title: {item['title']} \n {item['type']}:\n{item['content']}"
        
    for new_word in item['new_words']:
        output_response += f"\n[NEW WORD]: {new_word['word']}" #\n[MEANING]: {new_word['meaning']}" 
    
    input_prompts.append(input_prompt)
    output_responses.append(output_response)
    
    for q in item['exercises']:
        input_prompt = f"\n<task> Generate one {q['difficulty']} exercise question for the lesson.\n\n<lesson> Title: {item['title']} \n {item['type']}:\n{item['content']}"
        output_response = f"\n<exercise> [QUESTION]: {q['question']}\n[ANSWER]: {q['answer']}"  
        
        input_prompts.append(input_prompt)
        output_responses.append(output_response)

# tokenize inputs and outputs, and created a tokenized dataset with attention masks for training the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenized_inputs = tokenizer(
    input_prompts, 
    padding="max_length",  
    truncation=True, 
    max_length=1024,
    return_attention_mask=True
)
tokenized_outputs = tokenizer(
    output_responses,
    padding="max_length",
    truncation=True,
    max_length=1024,
    return_attention_mask=True
)
tokenized_dataset = Dataset.from_dict({ 
    "input_ids": tokenized_inputs["input_ids"],
    "attention_mask": tokenized_inputs["attention_mask"],
    "labels": tokenized_outputs["input_ids"],  
    "labels_attention_mask": tokenized_outputs["attention_mask"]
})

# load the pretrained model for training
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# freeze layers that need not be trained 
for param in model.transformer.wte.parameters():    # word token embedding layer
    param.requires_grad = False
for param in model.transformer.wpe.parameters():    # word positional embedding layer
    param.requires_grad = False
for param in model.transformer.h[:-2].parameters():  # transformer layers
    param.requires_grad = False

# for param in model.lm_head.parameters():      # output head of the model
#    param.requires_grad = False

# model.new_head = nn.Linear(model.config.hidden_size, tokenizer.vocab_size)
# model.new_head.requires_grad = True

# set training arguments 
# Note: the hyperparameters were set while training on a cpu
training_args = TrainingArguments(
    output_dir=f"./fine-tuned-{model_name}",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-8,
    save_steps=50,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=5,
    logging_strategy="steps",
    fp16=True,
    max_grad_norm=1.0
)

# train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

# save the model and tokenizer
model.save_pretrained(f"./fine-tuned-{model_name}")
tokenizer.save_pretrained(f"./fine-tuned-{model_name}")

