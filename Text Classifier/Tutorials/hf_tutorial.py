# Huggingface imports
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline

# Standard imports
import numpy as np

# Load the data
imdb = load_dataset("imdb")

# Select tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Tokenize the data
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True)  # max_length = window size (tensor-512)
tokenized_imdb = imdb.map(preprocess_function, batched=True)

# Select data collator (collector and organiser)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Select accuracy metric
accuracy = evaluate.load("accuracy")

# Calculate accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred  # get predictions and labels
    predictions = np.argmax(predictions, axis=1)  # calculate highest probability class (negative or positive sentiment)
    return accuracy.compute(predictions=predictions, references=labels)

# Select labels
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Select model and assign labels
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Model iteration count
c = 0
c += 1
path = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/bert_uncased_"+str(c)
print(path)

# Model training hyperparameter inputs
training_args = TrainingArguments(
    output_dir=path,  # path model stored
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Model training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Sample text
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# Text model
classifier = pipeline("sentiment-analysis", model=path)  # window_size = 512
print(classifier(text))