from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import os

# Load the dataset (assuming structured JSON dataset)
dataset = load_dataset('json', data_files='dataset.json')
print(len(dataset['train']))

from huggingface_hub import login
login(token = os.getenv('HF_TOKEN'))

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
    "labels": tokenized_outputs["input_ids"],  # Masked labels
    "labels_attention_mask": tokenized_outputs["attention_mask"]
})

model = AutoModelForCausalLM.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

for param in model.transformer.wte.parameters():
    param.requires_grad = False
for param in model.transformer.wpe.parameters():
    param.requires_grad = False
for param in model.transformer.h[:-2].parameters():
    param.requires_grad = False

# Freeze the output head
#for param in model.lm_head.parameters():
 #   param.requires_grad = False

#model.new_head = nn.Linear(model.config.hidden_size, tokenizer.vocab_size)
#model.new_head.requires_grad = True

training_args = TrainingArguments(
    output_dir="./fine-tuned-gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-8,
    save_steps=50,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=5,
    logging_strategy="steps",
    report_to="tensorboard",
    fp16=True,
    max_grad_norm=1.0
)


# Trainer for fine-tuning with detailed prompts
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")

