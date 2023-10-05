import os
import evaluate
import numpy as np
from datasets import Dataset
from torch.utils.data import random_split
from transformers import logging, AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer


# keep terminal clear of warnings/low-level messages
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# convert labels to integer id and vice versa
label2id = {
    "B-quantity": 0,
    "I-quantity": 1,
    "B-size": 2,
    "I-size": 3,
    "B-unit": 4,
    "I-unit": 5,
    "B-name": 6,
    "I-name": 7,
    "B-state": 8,
    "I-state": 9,
    "B-temp": 10,
    "I-temp": 11,
    "B-df": 12,
    "I-df": 13,
    "O": 14
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
    8: "B-state",
    9: "I-state",
    10: "B-temp",
    11: "I-temp",
    12: "B-df",
    13: "I-df",
    14: "O"
}

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
)


# convert token to readable and consistent formatting
def format_token(token):
    token = token.replace("½", "0.5")
    token = token.replace("⅓", "0.33")
    token = token.replace("⅔", "0.67")
    token = token.replace("¼", "0.25")
    token = token.replace("¾", "0.75")
    token = token.replace("⅕", "0.4")
    token = token.replace("⅖", "0.4")
    token = token.replace("⅗", "0.6")
    token = token.replace("⅘", "0.8")
    token = token.replace("⅞", "0.875")
    token = token.replace("-LRB-", "(")
    token = token.replace("-RRB-", ")")
    return token


# load and preprocess tsv data from a given file path
def preprocess_data(data_path):
    data = {"tokens": [], "labels": []}
    with open(data_path, "r") as f:
        lines = f.readlines()
        phrase = []
        labels = []
        last_label = ""

        # add a completed ingredient phrase to the dataset
        def append_phrase():
            nonlocal data, phrase, labels
            if labels and phrase:
                data["tokens"].append(phrase)
                data["labels"].append(labels)
            phrase = []
            labels = []

        # correctly label and add a token to the ingredient phrase
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
            if label != "O":
                tmp = tmp.lower()
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
    return Dataset.from_dict(data)


# tokenize the labels of a given dataset
def tokenize_labels(data):
    # tokenize the labels of a single example within the dataset
    def tokenize_label(example):
        tokenized = tokenizer(
            example["tokens"], padding=True, truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(example["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    # special tokens created by is_split_into_words labeled -100 (to be ignored)
                    # https://huggingface.co/docs/transformers/tasks/token_classification#preprocess
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    return data.map(tokenize_label, batched=True)


# split a given dataset into a training set and a testing set
def split_data(data, train_size=0.9):
    train_size = int(train_size * len(data))
    test_size = len(data) - train_size
    return random_split(data, [train_size, test_size])



# fine-tune a pre-trained model using tsv data from a given file path
def train(data_path, save_path="model"):
    data = preprocess_data(data_path)
    tokenized_data = tokenize_labels(data)
    train_data, test_data = split_data(tokenized_data)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")
    label_list = list(label2id.keys())

    # compute the precision, recall, f1 score, and accuracy of a given prediction
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

        results = seqeval.compute(
            predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=save_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        resume_from_checkpoint=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()


train("./data/train_full.tsv", "model")