

def pred(topic, iteration):
    # DOESN'T WORK -> had to just reboot VS code. Try this DEVICE = "cuda" if torch.cuda.is_available() else "cpu".
    # # Hot fix for running out of memory
    # import torch
    # torch.cuda.empty_cache()


    # MS: create training set, evaluation set and validation set
    import sys
    import pandas as pd
    import csv

    dataset_name = "/home/azureuser/cloudfiles/code/Users/Omololu.Makinde/Llama_tutorial/data/consultation2.csv"
    dataset = []

    with open(dataset_name, encoding='utf-8-sig') as FID:
        csvReader = csv.DictReader(FID, delimiter="\t")
        for key, row in enumerate(csvReader): 
            # print(key, row)
            dataset.append({
                "label" : row['Topic'].strip().lower().replace('\n', ''),
                "text" : row['Full summary of comment'],
                "summary" : row['One-line summary']
            })

    df_dataset = pd.DataFrame(dataset)
    df_dataset.drop(columns =['summary'], inplace=True)  # don't need this for now
    # display(df_dataset)



    df_dataset_count = df_dataset['label'].value_counts()
    df_dataset['label'].value_counts()




    keep_rows = list(df_dataset_count[df_dataset_count > 10].index)
    df_dataset = df_dataset[df_dataset['label'].isin(keep_rows)]
    # display(df_dataset['label'].value_counts())




    import numpy as np

    # add in our classifier labels
    df_dataset['label'] = np.where(np.array(df_dataset['label']) == topic, 1, 0)
    df_dataset['label'].value_counts(normalize=False)



    # train and test data
    class_len = len(df_dataset[df_dataset['label'] == 1])  # find how many values we can take and still have a balanced class
    class_0_data = df_dataset[df_dataset.label.eq(0)].sample(class_len) 
    class_1_data = df_dataset[df_dataset.label.eq(1)].sample(class_len)
    train_test_data = pd.concat([class_0_data, class_1_data])  # 50/50 class split
    # display(train_test_data)

    # evaluation data
    eval_data = df_dataset.drop(train_test_data.index)  # put the rest into an evaluation set we can play with
    eval_data



    from datasets import Dataset
    from transformers import AutoTokenizer

    huggingface_data = Dataset.from_pandas(train_test_data, preserve_index=False)  # don't include pandas index

    pretrained_model_name = "distilbert/distilbert-base-uncased"  # This is our base model

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    def proc_data(data):
        return tokenizer(data['text'], max_length=512, padding=True, truncation=True)

    tokenized_data = huggingface_data.map(proc_data, batched=True)  # advantage of ".map" is we can parallel process data in batches (i think)
    # print(tokenized_data['text'] == huggingface_data['text'])  # SHOULDN'T THIS NOT BE TRUE??? ESPECIALLY IF I CHANGE MAX_LENGTH TO BE 1 OR SOMETHING

    split_tokenized_hugginface_data = tokenized_data.train_test_split(test_size=0.10)  # 85/15 train/test split
    print(split_tokenized_hugginface_data)



    import os
    import shutil

    def clear_folder_except(folder_path, exclude_folder):
        # Iterate over the items in the directory
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            # Skip the exclude folder
            if item_name == exclude_folder:
                continue
            # Check if the item is a directory
            if os.path.isdir(item_path):
                # If it is a directory, remove it and its contents
                shutil.rmtree(item_path)
            else:
                # If it is a file, remove it
                os.remove(item_path)




    # -HOT FIX TO STOP MODEL TRAINING BREAKING----------------------------------------------------------------------------------------------------------- #
    model_output_path = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/Mod_" + str(iteration)
    # clear_folder_except(model_output_path, "runs")
    # --------------------------------------------------------------------------------------------------------------------------------------------------- #


    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
    from transformers import DataCollatorWithPadding
    import evaluate


    # Select accuracy metric
    evaluation_metrics = ["accuracy", "f1", "precision", "recall"]
    accuracy = evaluate.combine(evaluation_metrics)# evaluate.load("accuracy")

    # Use accuracy to determine which class is the most likely prediction
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred  # get predictions and labels
        predictions = np.argmax(predictions, axis=1)  # calculate highest probability class (negative or positive sentiment)
        return accuracy.compute(predictions=predictions, references=labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative": 0, "Positive": 1}

    # Define device
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # This will hopefully force use of GPU and stop killing the VM's memory on the CPU


    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name,
                                                                id2label=id2label,
                                                                label2id=label2id)

    model.to(device)  # mount model onto GPU

    # How we input training arguments into the model

    training_args = TrainingArguments(
        output_dir=model_output_path,  # path model stored
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps= 25 # logs made every X batches. So smaller log means more information recorded (see log in table but also more computational and memory requirements)
        )


    # Model training arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_tokenized_hugginface_data["train"],#.select(range(200)),
        eval_dataset=split_tokenized_hugginface_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Stop training if no improvement after 3 consecutive epochs

    )

    # Train model
    print(iteration, topic)
    trainer.train()


    # Save the model
    best_model_path = model_output_path + "/Best"
    trainer.save_model(best_model_path)

    # get best model scores
    print(trainer.evaluate())



    # Parse the pdf
    from tika import parser

    pdf_filepath = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Overview.pdf"
    parsed_file = parser.from_file(pdf_filepath)['content']
    # print(parsed_file)


    # Split pdf into chunks
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-small-Embedding-v0")  # get embeddings model
    text_splitter = SemanticChunker(embedding)  # apply the model to the type of split we want to perform (a semantic split)
    chunks = text_splitter.create_documents([parsed_file])


    output_folder = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Output Data"
    # print(len(chunks), chunks[0])

    import json
    from collections import defaultdict

    dict_chunks = defaultdict(list)
    for par in chunks:
        dict_chunks[par.page_content]=[]
    outfile = output_folder + "/_" + str(iteration) + "_infer_output.json"
    with open(outfile, "w") as outfile: 
        json.dump(dict_chunks, outfile, indent = 4)
    print(outfile)


    outfile = output_folder + "/_" + str(iteration) + "_infer_output.json"
    with open(str(outfile), 'r') as empt_par:
        strored_chunk = json.load(empt_par)
    strored_chunk


    from transformers import pipeline

    def inference_pipeline(model_path, max_length=512):
        pipe = pipeline("text-classification", model=model_path, max_length=max_length, truncation=True)
        return pipe

    infer = inference_pipeline(best_model_path)


    # test
    print(infer("Governance"))
    print(infer("Bye"))


    # try on our data
    infer(str(strored_chunk)[0].replace("\n\n", ""))


    # Now do it properly
    res = {}

    for i in list(strored_chunk):
        # res[i] = {"topic" : topic}
        val = infer(i.replace('\n\n', ''))[0]
        val['class'] = topic
        res[i] = val
    res


    tagged_output = output_folder + "/_" + str(iteration) + "_infer_output.json"
    with open(tagged_output, "w") as tagged_par: 
        json.dump(res, tagged_par, indent = 4)



    return

import time
s = time.time()
iteration = 0
Topics = ["approach to the codes",                                 
"automated content moderation (user to user)",           
"governance and accountability"]

for topic in Topics:
    iteration+=1
    pred(topic, iteration)

time.time()-s