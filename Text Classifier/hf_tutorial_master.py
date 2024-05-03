### Error seems to be coming from Shape mismatch -> check "tf_dataset()" part ###



from datasets import load_dataset

imdb = load_dataset("imdb")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"])
tokenized_imdb = imdb.map(preprocess_function, batched=True)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
import evaluate

accuracy = evaluate.load("accuracy")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
from transformers import create_optimizer
# import tensorflow as tf

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
tf_train_set = model.prepare_tf_dataset(
    tokenized_imdb["train"],
    batch_size=16,
    shuffle=True,
    tokenizer=tokenizer
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_imdb["test"],
    batch_size=16,
    shuffle=True,
    tokenizer=tokenizer
)


model.compile(optimizer=optimizer)  # No loss argument!
# from transformers.keras_callbacks import KerasMetricCallback

# metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
# from transformers.keras_callbacks import PushToHubCallback

# push_to_hub_callback = PushToHubCallback(
#     output_dir="my_awesome_model",
#     tokenizer=tokenizer,
# )
# callbacks = [metric_callback, push_to_hub_callback]
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3)
