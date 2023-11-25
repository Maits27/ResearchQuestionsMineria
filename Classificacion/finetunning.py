from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



LR = 2e-5
EPOCHS = 1
BATCH_SIZE = 8
MODEL = "cardiffnlp/twitter-xlm-roberta-base" # use this to finetune the language model
MAX_TRAINING_EXAMPLES = -1 # set this to -1 if you want to use the whole training set
TAMAÑO_DEL_DATASET = 10000


dataPathOnDrive = "./formated/"
dataToWrite = "./results/"

data = pd.read_csv(f'{dataPathOnDrive}/Suicide_Detection_10000.csv')
assert len(data) == TAMAÑO_DEL_DATASET

# Train: 75%; Dev: 15%; Test: 10%
train, test_val = train_test_split(data, test_size=0.25, random_state=42)
val, test = train_test_split(test_val, test_size=0.4, random_state=42)

# Generamos la estructura de datos apropiada para que funcione el código del Notebook original
dataset_dict = {}
dataset_dict['train'] = {}
dataset_dict['train']['text'] = [text for text in train['text']]
dataset_dict['train']['labels'] = [0 if label == 'suicide' else 1 for label in train['class']]
assert len(dataset_dict['train']['text']) == TAMAÑO_DEL_DATASET*0.75
assert len(dataset_dict['train']['labels']) == TAMAÑO_DEL_DATASET*0.75
print(f'Train set texts: {len(dataset_dict["train"]["text"])}; labels: {len(dataset_dict["train"]["labels"])}')

dataset_dict['val'] = {}
dataset_dict['val']['text'] = [text for text in val['text']]
dataset_dict['val']['labels'] = [0 if label == 'suicide' else 1 for label in val['class']]
assert len(dataset_dict['val']['text']) == TAMAÑO_DEL_DATASET*0.15
assert len(dataset_dict['val']['labels']) == TAMAÑO_DEL_DATASET*0.15
print(f'Val set texts: {len(dataset_dict["val"]["text"])}; labels: {len(dataset_dict["val"]["labels"])}')

dataset_dict['test'] = {}
dataset_dict['test']['text'] = [text for text in test['text']]
dataset_dict['test']['labels'] = [0 if label == 'suicide' else 1 for label in test['class']]
assert len(dataset_dict['test']['text']) == TAMAÑO_DEL_DATASET*0.10
assert len(dataset_dict['test']['labels']) == TAMAÑO_DEL_DATASET*0.10
print(f'Test set texts: {len(dataset_dict["test"]["text"])}; labels: {len(dataset_dict["test"]["labels"])}')


if MAX_TRAINING_EXAMPLES > 0:
  dataset_dict['train']['text']=dataset_dict['train']['text'][:MAX_TRAINING_EXAMPLES]
  dataset_dict['train']['labels']=dataset_dict['train']['labels'][:MAX_TRAINING_EXAMPLES]



tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, model_max_length=512) # Model max length is not set...

train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True)
val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True)
test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, padding=True)
print(train_encodings[:2])
print(train_encodings)


train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])

training_args = TrainingArguments(
    evaluation_strategy = 'steps',            # when to evaluate. "steps": Evaluation is done (and logged) every eval_steps (or logging_steps=10 if eval_steps missing); "no": No evaluation is done during training; "epoch": Evaluation is done at the end of each epoch;
    output_dir='./results',                   # output directory
    num_train_epochs=EPOCHS,                  # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
    warmup_steps=100,                         # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                        # strength of weight decay
    logging_dir='./logs',                     # directory for storing logs
    logging_steps=10,                         # when to print log (and to evaluate if evaluation_strategy = 'steps')
    load_best_model_at_end=True,              # load or not best model at the end
)

num_labels = len(set(dataset_dict["train"]["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

trainer = Trainer(
    model=model,                              # the instantiated 🤗 Transformers model to be trained
    args=training_args,                       # training arguments, defined above
    train_dataset=train_dataset,              # training dataset
    eval_dataset=val_dataset                  # evaluation dataset
)

trainer.train()

trainer.save_model(dataToWrite + 'best_model') # save best model

test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_raw, axis=-1)
print(classification_report(test_labels, test_preds, digits=3))