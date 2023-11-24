from transformers import AutoModel, AutoTokenizer
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
BATCH_SIZE = 16
MODEL = "cardiffnlp/twitter-xlm-roberta-base" # use this to finetune the language model
MAX_TRAINING_EXAMPLES = -1 # set this to -1 if you want to use the whole training set


model_name = "cardiffnlp/twitter-xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Imprime las dimensiones originales de la capa de embeddings
print("Dimensiones originales:", model.embeddings.word_embeddings.weight.shape)

# Ajusta las dimensiones de la capa de embeddings
new_embedding_dimensions = 6  # Elige la dimensión que desees
model.resize_token_embeddings(new_embedding_dimensions)

# Imprime las nuevas dimensiones de la capa de embeddings
print("Nuevas dimensiones:", model.embeddings.word_embeddings.weight.shape)

# Ahora puedes continuar con el proceso de finetuning
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