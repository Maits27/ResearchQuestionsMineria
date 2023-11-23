from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

import numpy as np
from sklearn.metrics import classification_report

# 0. Setting parameters
LR = 2e-5
EPOCHS = 1
BATCH_SIZE = 32
MODEL = "cardiffnlp/twitter-xlm-roberta-base" # use this to finetune the language model
MAX_TRAINING_EXAMPLES = -1 # set this to -1 if you want to use the whole training set

# 1. Loading (example) data
pathToRead = "./formated/"

dataset_dict = {}
for i in ['train','val','test']:
  dataset_dict[i] = {}
  for j in ['text','labels']:
    dataset_dict[i][j] = open(f"{pathToRead}{i}_{j}.txt").read().split('\n')
    if j == 'labels':
      dataset_dict[i][j] = [int(x) for x in dataset_dict[i][j]]

if MAX_TRAINING_EXAMPLES > 0:
  dataset_dict['train']['text']=dataset_dict['train']['text'][:MAX_TRAINING_EXAMPLES]
  dataset_dict['train']['labels']=dataset_dict['train']['labels'][:MAX_TRAINING_EXAMPLES]

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True)
val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True)
test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, padding=True)

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

train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])

# 2. Fine-tuning
training_args = TrainingArguments(
    evaluation_strategy = 'steps',
    output_dir='./results',                   # output directory
    num_train_epochs=EPOCHS,                  # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
    warmup_steps=100,                         # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                        # strength of weight decay
    logging_dir='./logs',                     # directory for storing logs
    logging_steps=10,                         # when to print log
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
trainer.save_model("./results/best_model") # save best model

# 3. Evaluate on test set
test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_raw, axis=-1)
print(classification_report(test_labels, test_preds, digits=3))

