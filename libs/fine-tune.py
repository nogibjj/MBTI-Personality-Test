#!/usr/bin/env python

"""
Fine Tuning Example with HuggingFace

Based on official tutorial
"""
import os
from datasets.load import load_from_disk
from transformers import AutoTokenizer
from evaluate import load
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

MODEL_PATH = "bert-base-uncased"
TOKENIZER_PATH = "bert-base-uncased"
CHECK_POINT_PATH = "/workspaces/MBTI-Personality-Test/mbti-classification/checkpoint-45000"
OUTPUT_DIR = "/workspaces/MBTI-Personality-Test/mbti-classification"
DATA_PATH = "/workspaces/MBTI-Personality-Test/tokenized_datasets"
REPO_DIR = "mbti-classification-bert-base-uncased"

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=265)


# Load the dataset
if not os.path.exists(DATA_PATH):
    dataset = load_dataset("Shunian/kaggle-mbti-cleaned")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk(DATA_PATH)
else:
    tokenized_datasets = load_from_disk(DATA_PATH, keep_in_memory=True)


# Load the model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=16, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

metric = load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# can use if needed to reduce memory usage and training time
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)


training_args = TrainingArguments(  output_dir=OUTPUT_DIR, 
                                    evaluation_strategy="epoch", 
                                    save_strategy="steps",
                                    save_steps=5000,
                                    save_total_limit=5, 
                                    per_device_train_batch_size=24, 
                                    per_device_eval_batch_size=24, 
                                    learning_rate=2e-5, 
                                    num_train_epochs=5, 
                                    weight_decay=0.01,
                                    lr_scheduler_type='cosine',
                                    push_to_hub=True,
                                    hub_model_id=REPO_DIR,
                                    hub_strategy="every_save")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=CHECK_POINT_PATH) # train the model resume_from_checkpoint=CHECK_POINT_PATH
trainer.push_to_hub(REPO_DIR) 
model.push_to_hub(REPO_DIR)
tokenizer.push_to_hub(REPO_DIR)