import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
def chunks(lista, batch_size):
    for i in range(0, len(lista), batch_size):
        yield lista [i:i+batch_size]

# MODEL_PATH = 'AingeruBeOr/SuicideDetectionOnTweets'
MODEL_PATH = 'Maits27/TextSentimentBasedForSuicide'
# MODEL_PATH = 'cardiffnlp/twitter-xlm-roberta-base'
# MODEL_PATH = 'Maits27/OnlySentimentBased'
# MODEL_PATH = 'Maits27/SuicideDetectionWithEmotion' #SinClase
TOKENIZER = 'cardiffnlp/twitter-xlm-roberta-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 16


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True, model_max_length=512) # Model max length is not set...
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)

test = pd.read_csv('formated/Suicide_Detection_Test_5000.csv')

texts = test['text'].to_list()
tiny_texts = test['text'][:5000].to_list()

outputs = []
for batch in tqdm(chunks(tiny_texts, BATCHSIZE), total=int(len(tiny_texts)/BATCHSIZE)):
    test_tokens = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=512, add_special_tokens=True).to(DEVICE)
    with torch.no_grad():  # Disabling gradient calculation is useful for inference, when you are sure that you will not update weigths
        outputs.append(model(**test_tokens)[0].detach().cpu())

outputs=torch.cat(outputs)

correct = 0
predicciones = {}
probability_to_class_1 = []
probability_to_class_0 = []
for index, output in enumerate(outputs):
    probability_to_class_1.append(output[1])
    probability_to_class_0.append(output[0])
    if output[0] > output[1]: predicted = 'suicide'
    else: predicted = 'non-suicide'

    print(f'{test["text"][index][:30]}; real: {test["class"][index]}; probabilidad: {predicted}')

    if test["class"][index] == predicted: correct += 1
    predicciones[index] = predicted

np.save('../Predicciones/probability_to_class_1.npy', np.array(probability_to_class_1))
np.save('../Predicciones/probability_to_class_0.npy', np.array(probability_to_class_0))

with open(f'../Predicciones/Predicciones{len(predicciones)}.json', 'w', encoding='utf-8') as json_file:
    json.dump(predicciones, json_file, indent=2, ensure_ascii=False)

