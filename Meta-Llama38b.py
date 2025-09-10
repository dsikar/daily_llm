# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

# Load model directly
#from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
