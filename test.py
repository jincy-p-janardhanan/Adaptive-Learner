from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2" # update if a different model is used for training
device = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available

content_prompt = "<task> Generate a lesson for drama at difficulty level 1." # modify to try different prompts

# load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"./fine-tuned-{model_name}")
model = AutoModelForCausalLM.from_pretrained(f"./fine-tuned-{model_name}")

# tokenize the prompt
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(content_prompt):
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

    # Decode and return the generated response
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_response(content_prompt))

