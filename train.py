import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, logging

logging.set_verbosity_error()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

label2id = {
    "B-quantity": 0,
    "I-quantity": 1,
    "B-size": 2,
    "I-size": 3,
    "B-unit": 4,
    "I-unit": 5,
    "B-name": 6,
    "I-name": 7,
    "B-df": 8,
    "I-df": 9,
    "B-state": 10,
    "I-state": 11,
    "O": 12
}

id2label = {
    0: "B-quantity",
    1: "I-quantity",
    2: "B-size",
    3: "I-size",
    4: "B-unit",
    5: "I-unit",
    6: "B-name",
    7: "I-name",
    8: "B-df",
    9: "I-df",
    10: "B-state",
    11: "I-state",
    12: "O"
}


def format_token(ingredient):
    ingredient = ingredient.replace("½", "0.5")
    ingredient = ingredient.replace("⅓", "0.33")
    ingredient = ingredient.replace("⅔", "0.67")
    ingredient = ingredient.replace("¼", "0.25")
    ingredient = ingredient.replace("¾", "0.75")
    ingredient = ingredient.replace("⅕", "0.4")
    ingredient = ingredient.replace("⅖", "0.4")
    ingredient = ingredient.replace("⅗", "0.6")
    ingredient = ingredient.replace("⅘", "0.8")
    ingredient = ingredient.replace("⅞", "0.875")
    ingredient = ingredient.replace("-LRB-", "(")
    ingredient = ingredient.replace("-RRB-", ")")
    return ingredient


def preprocess():
    dataset = {"tokens": [], "labels": []}
    with open("./data/train_full.tsv", "r") as f:
        lines = f.readlines()
        phrase = []
        labels = []
        last_label = ""

        def append_phrase():
            nonlocal dataset, phrase, labels
            if labels and phrase:
                dataset["tokens"].append(phrase)
                dataset["labels"].append(labels)
            phrase = []
            labels = []

        def append_token(token, label):
            nonlocal phrase, labels, last_label
            tmp = label
            if label != "O":
                label = label.lower()
                if label == last_label:
                    label = "I-" + label
                else:
                    label = "B-" + label
            phrase.append(token)
            labels.append(label2id[label])
            if label != "O": tmp = tmp.lower() 
            last_label = tmp

        for line in lines:
            line = line.strip().strip("\n")
            if not line:
                append_phrase()
            else:
                token, label = line.split("\t")
                token = format_token(token)
                token = token.lower()
                if len(token.split()) > 1:
                    tokens = token.split()
                    for i in range(len(tokens)):
                        append_token(tokens[i], label)
                else:
                    append_token(token, label)
        append_phrase()
    return dataset


dataset = preprocess()


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # special tokens created by is_split_into_words labeled -100 (to be ignored)
                # https://huggingface.co/docs/transformers/tasks/token_classification#preprocess
                label_ids.append(-100) 
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


from datasets import Dataset

dataset = Dataset.from_dict(dataset)
inputs = dataset.map(tokenize_and_align_labels, batched=True)




from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

import evaluate

seqeval = evaluate.load("seqeval")

import numpy as np

label_list = list(label2id.keys())


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=13, id2label=id2label, label2id=label2id
)



training_args = TrainingArguments(
    output_dir="model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=inputs,
    eval_dataset=inputs,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()