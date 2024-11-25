from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-gpt2")
model.to(device)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-gpt2")

# Set the pad token to eos token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Example prompt for generation
content_prompt = "<task> Generate a lesson for drama at difficulty level 1."
# Tokenize the input prompt with attention mask
inputs = tokenizer(content_prompt, return_tensors="pt", padding=True)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

# Generate output from the fine-tuned model
output = model.generate(
    input_ids, 
    attention_mask=attention_mask,
    max_length=400, 
    temperature=0.7,  
    do_sample=True,
    top_k=50,
    top_p=0.95
)

# Decode and print the generated text
print(tokenizer.decode(output[0], skip_special_tokens=True))

