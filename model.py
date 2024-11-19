from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load the dataset (assuming structured JSON dataset)
dataset = load_dataset('json', data_files='dataset.json')

from huggingface_hub import login
login(token = os.getenv('HF_TOKEN'))

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Tokenization function with enhanced prompts
def tokenize_function(examples):
    inputs = []
    outputs = []

    # Iterate over each example using a loop
    for idx in range(len(examples['id'])):
        # Access individual items using indices
        item = {key: examples[key][idx] for key in examples.keys()}
        
        # Prepare input prompt based on content type and difficulty
        input_prompt = f"[TASK] Generate a {item['type']} lesson at  {item['difficulty']} level.\n"
                
        # Specify the main content
        input_prompt += "[TITLE]: \n[PROLOGUE]: \n[CONTENT]"
        output_response = f"\n [TITLE]: {item['title']} \n[PROLOGUE]: {item['prologue']} \n[CONTENT]: {item['content']}"
        chapter_content = output_response
        
        # Generate 'New Words' section
        input_prompt += f"\n[TASK] List new words with meanings for this content.\n"
        for new_word in item['new_words']:
            output_response += f"\n[NEW WORD]: {new_word['word']}\n[MEANING]: {new_word['meaning']}"
        
        # Exercise questions with difficulty levels
        input_prompt += f"\n[TASK] Generate exercise questions for \n[CHAPTER-START]\n{chapter_content}\n[CHAPTER-END] at varying difficulty levels.\n"
        for q in item['exercises']:
            input_prompt += f"\n[DIFFICULTY]: {q['difficulty']}"
            output_response += f"\n[QUESTION]: {q['question']}\n[ANSWER]: {q['answer']}"
        
        inputs.append(input_prompt)
        outputs.append(output_response)

    # Assign the EOS token to the pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize inputs and outputs
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True)
    tokenized_outputs = tokenizer(outputs, padding="max_length", truncation=True)
    
    # Return a dictionary, combining the tokenized inputs and outputs
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_outputs["input_ids"],  # Using input_ids as labels for outputs
    }

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the LLaMA model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Make sure you're using a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-llama-enhanced",
    num_train_epochs=5,
    per_device_train_batch_size=1,  # Reduced batch size for large models
    gradient_accumulation_steps=4,  # To handle memory with smaller batch sizes
    save_steps=200,
    save_total_limit=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=20,
    fp16=True,  # Mixed precision training for efficiency
)

# Trainer for fine-tuning with detailed prompts
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

# Start fine-tuning
trainer.train()

# Example of generating content using the fine-tuned model
content_prompt = "[TASK] Generate a prose content at Intermediate level.\n[CONTEXT]: The topic is about a morning adventure."
input_ids = tokenizer(content_prompt, return_tensors="pt").input_ids.to(device)
output = model.generate(input_ids, max_length=400, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Example of generating new words with meanings
new_words_prompt = "[TASK] List new words with meanings from a prose content at Intermediate level.\n[CONTENT]: The leaves rustled gently as the morning breeze swirled around Jamie."
input_ids = tokenizer(new_words_prompt, return_tensors="pt").input_ids.to(device)
output = model.generate(input_ids, max_length=150, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
