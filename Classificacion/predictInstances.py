from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

MODEL_PATH = './models/'
TOKENIZER = 'cardiffnlp/twitter-xlm-roberta-base'

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True, model_max_length=512) # Model max length is not set...
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
print(model)

test = pd.read_csv('formated/test.csv')

texts = test['text'].to_list()
tiny_texts = test['text'][:1000].to_list()

test_tokens = tokenizer(tiny_texts, truncation=True, padding=True, return_tensors='pt', max_length=512, add_special_tokens=True)


with torch.no_grad(): # Disabling gradient calculation is useful for inference, when you are sure that you will not update weigths
    outputs = model(**test_tokens)[0]

correct = 0
for index, output in enumerate(outputs):
    if output[0] > output[1]: predicted = 'suicide' 
    else: predicted = 'non-suicide'

    print(f'{test["text"][index][:30]}; real: {test["class"][index]}; probabilidad: {predicted}')

    if test["class"][index] == predicted: correct += 1

print(correct)
