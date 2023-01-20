import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import warnings
warnings.filterwarnings("ignore")
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
import os
if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')
import torch

while True:
	print()
	input_text = input("Enter your question: ")
	
	input_ids = tokenizer(input_text, return_tensors="pt").input_ids
	attention_mask = torch.ones(input_ids.shape)
	pad_token_id = tokenizer.eos_token_id
    
	tf.random.set_seed(0)

	sample_outputs = model.generate(
	    input_ids,
	    attention_mask=attention_mask,
	    pad_token_id=pad_token_id,
	    do_sample=True, 
	    max_length=150, 
	    top_k=50, 
	    top_p=0.95, 
	    num_return_sequences=7
	)

	
	for i, sample_output in enumerate(sample_outputs):
	  print("-------------------------------------------------------------------------------------------------------------------------------------------")
	  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
	  print()


