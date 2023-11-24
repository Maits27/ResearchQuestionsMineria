from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import time

start_time = time.time()
print(torch.device('mps'))
print(torch.backends.mps.is_available()) # Is MPS even available? macOS 12.3+
print(torch.backends.mps.is_built()) # Is MPS even available? macOS 12.3+
print(torch.cuda.is_available())
DEVICE = 'mps' # Para usar la GPU del MAC, usa el framework Metal de Apple. 'cuda' si fuese una gráfica de NVIDIA
TOKENIZER = 'cardiffnlp/twitter-xlm-roberta-base'
MODEL_PATH = 'AingeruBeOr/SuicideDetectionOnTweets' # Try with cardiffnlp/twitter-xlm-roberta-base to check performance

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True, model_max_length=512) # Model max length is not set...
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to('mps')
print('✅ Model loaded successfuly') # print(model)

test = pd.read_csv('../Datasets/Suicide_Detection100.csv')

texts = test['text'].to_list()
#tiny_texts = test['text'][:1000].to_list()

test_tokens = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=512, add_special_tokens=True).to('mps')
print('✅ Tokenization completed successfuly')

with torch.no_grad(): # Disabling gradient calculation is useful for inference, when you are sure that you will not update weigths
    outputs = model(**test_tokens)[0]
print('✅ Clasification completed successfuly')
finish_time = time.time()

predicted_y = []
for index, output in enumerate(outputs):
    if output[0] > output[1]: predicted, predicted_int = 'suicide', 0
    else: predicted, predicted_int = 'non-suicide', 1
    predicted_y.append(predicted_int)

    #print(f'{test["text"][index][:30]}; real: {test["class"][index]}; probabilidad: {predicted}')

real_y = [0 if value == 'suicide' else 1 for index, value in test['class'].items()]

# Metrics
print('Execution time:', finish_time-start_time)
print(' --- Metrics ---')
accuracy = accuracy_score(real_y, predicted_y)
confusion = confusion_matrix(real_y, predicted_y)
report = classification_report(real_y, predicted_y)
kappa = cohen_kappa_score(real_y, predicted_y)

print('Accuracy', accuracy)
print('Confusion matrix\n', confusion)
print(report)
print('Kappa:', kappa)
