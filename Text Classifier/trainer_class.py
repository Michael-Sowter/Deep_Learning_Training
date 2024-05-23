import sys
import time
import os
import pandas as pd
import csv
import numpy as np
import logging

import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer



class TextClassifierTrainer:

    def __init__(self, dataset_name : str, topic : str, learning_rate : float, epochs : int, pretrained_model_name : str, weight_decay : float):
        self.dataset_name = dataset_name
        self.topic = topic
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

    def create_dataset(self):
        dataset = []
        with open(self.dataset_name, encoding='utf-8-sig') as FID:
            csvReader = csv.DictReader(FID, delimiter="\t")
            for key, row in enumerate(csvReader): 
                # print(key, row)
                dataset.append({
                    "label" : row['Topic'].strip().lower().replace('\n', ''),
                    "text" : row['Full summary of comment'],
                    "summary" : row['One-line summary']
                })

        df_dataset = pd.DataFrame(dataset)
        df_dataset.drop(columns =['summary'], inplace=True)  # don't need this field for now
        df_dataset_count = df_dataset['label'].value_counts()
        keep_rows = list(df_dataset_count[df_dataset_count > 10].index)
        df_dataset = df_dataset[df_dataset['label'].isin(keep_rows)]
        # Add in our classifier labels
        df_dataset['label'] = np.where(np.array(df_dataset['label']) == self.topic, 1, 0)
        df_dataset['label'].value_counts(normalize=False)
        # Train and test data
        class_len = len(df_dataset[df_dataset['label'] == 1])  # find how many values we can take and still have a balanced class
        class_0_data = df_dataset[df_dataset.label.eq(0)].sample(class_len) 
        class_1_data = df_dataset[df_dataset.label.eq(1)].sample(class_len)
        train_test_data = pd.concat([class_0_data, class_1_data])  # 50/50 class split
        # Evaluation data
        eval_data = df_dataset.drop(train_test_data.index)  # put the rest into an evaluation set we can play with
        # Transform data into a hugging face compatible dataset for our models
        huggingface_data = Dataset.from_pandas(train_test_data, preserve_index=False)  # don't include pandas index

            
        def proc_data(data):
            return self.tokenizer(data['text'], max_length=512, padding=True, truncation=True)

        tokenized_data = huggingface_data.map(proc_data, batched=True)  # advantage of ".map" is we can parallel process data in batches (i think)
        split_tokenized_hugginface_data = tokenized_data.train_test_split(test_size=0.10)
        
        return split_tokenized_hugginface_data, eval_data
    
    def train_model(self, dataset):

        # Select accuracy metric
        evaluation_metrics = ["accuracy", "f1", "precision", "recall"]
        accuracy = evaluate.combine(evaluation_metrics)# evaluate.load("accuracy")

        # Use accuracy to determine which class is the most likely prediction
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred  # get predictions and labels
            predictions = np.argmax(predictions, axis=1)  # calculate highest probability class (negative or positive sentiment)
            return accuracy.compute(predictions=predictions, references=labels)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        id2label = {0: "Negative", 1: "Positive"}
        label2id = {"Negative": 0, "Positive": 1}

        # Load model and mount to GPU    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # This will hopefully force use of GPU and stop killing the VM's memory on the CPU
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, id2label=id2label, label2id=label2id)
        model.to(device)  # mount model onto GPU

        # How we input training arguments into the model
        training_args = TrainingArguments(
            output_dir=model_output_path,  # path model stored
            learning_rate=self.learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            push_to_hub=False,
            metric_for_best_model = 'f1',
            load_best_model_at_end=True  # Do this so we have the training_state.json of our best model
            # logging_steps= 25 # logs made every X batches. So smaller log means more information recorded (see log in table but also more computational and memory requirements)
            )

        # Model training arguments
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],#.select(range(10)),  # can limit our data input with select
            eval_dataset=dataset["test"],#.select(range(1)),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Stop training if no improvement after 3 consecutive epochs
        )

        return trainer


if __name__ != '__main__':
    sys.exit("Check you are running the file correctly!")


# Initiate logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="trainer.log", level=logging.INFO, filemode="w")
logger.info("Script Started")
s = time.time()

dataset_name = "/home/azureuser/cloudfiles/code/Users/Omololu.Makinde/Llama_tutorial/data/consultation2.csv"
pretrained_model_name = "distilbert/distilbert-base-uncased"  # This is our base model
Topics = ["approach to the codes", "automated content moderation (user to user)", "governance and accountability"]

# Model tuning parameters to try varying
learning_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001] #2*10**-5  # suggested range: 10^-1 to 10^-5
epochs = 10  # suggested range: 10 to 100
weight_decay_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]  #0.01  # suggested range: 10^-6 to 10^-1


for topic in Topics:
    logger.info(f"Topic: {topic}")
    for learning_rate in learning_rate_list:
        weight_decay = 0.01
        logger.info(f"Training new model... ")
        logger.info(f"PARAMETERS \t Learning Rate: {learning_rate} \t Weight Decay: {weight_decay} \t Epochs: {epochs}")
        logger.info("Script Ended")
        model_output_path = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/"+ topic + "/Batch/lr:"+str(learning_rate) + "_wd:" + str(weight_decay)
        tct = TextClassifierTrainer(dataset_name, topic, learning_rate, epochs, pretrained_model_name, weight_decay)
        dataset, eval_data = tct.create_dataset()
        trainer = tct.train_model(dataset)
        
        # Train model
        print(topic)
        train_result = trainer.train()

        # Save the model 
        best_model_path = model_output_path + "/Best"
        trainer.save_model(best_model_path)

        # Get best model scores
        print(trainer.evaluate())

    break  # stop at first topic

print("run time:", time.time()-s, "s")
logger.info("Script Ended")
logger.info("run time: " + str(round(time.time()-s, 1)) + "s")



# for topic in Topics:
#     learning_rate = 10**-5
#     weight_decay = 0.01
#     model_output_path = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/"+ topic + "/no_train"
#     tct = TextClassifierTrainer(dataset_name, topic, learning_rate, epochs, pretrained_model_name, weight_decay)
#     dataset, eval_data = tct.create_dataset()
#     trainer = tct.train_model(dataset)
        
#     # Train model
#     print(topic)

#     # Get best model scores
#     print(trainer.evaluate())

#     break  # stop at first topic

# print("run time:", time.time()-s, "s")

