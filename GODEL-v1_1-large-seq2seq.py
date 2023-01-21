import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")

def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, do_sample=True, max_length=150, top_k=50, top_p=0.95, num_return_sequences=7)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

instruction = f'Instruction: given a dialog context, you need to response empathically.'

if os.name == 'nt':
    os.system('cls')
else:
    os.system('clear')

while True:
    knowledge = ''

    question = input("Enter your question: ")
    dialog = [question]
    print("-------------------------------------------------------------------------------------------------------------------------------------------")

    for i in range(7):
        response = generate(instruction, knowledge, dialog)
        print(response)
        print()
           
