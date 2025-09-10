
# https://huggingface.co/google/gemma-2-9b?library=transformers
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-2-9b")     # Load model directly
#from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
#model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")

pipe = pipeline("text-generation", model="google/gemma-2-9b")
prompt = "What year was Jesus Christ born?"
result = pipe(prompt, max_new_tokens=50)
print("Pipeline result:")
print(result[0]["generated_text"])

#inputs = tokenizer(prompt, return_tensors="pt")
#outputs = model.generate(**inputs, max_new_tokens=50)
#generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print("\nManual result:")
#print(generated_text)
