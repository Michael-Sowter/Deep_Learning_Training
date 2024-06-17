from transformers import BartForSequenceClassification, BartTokenizer
from transformers import AutoTokenizer, AutoModel
import logging
import torch
import time
import pandas as pd
from torch.nn import functional as F
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import csv
from pathlib import Path


def make_data(dataset_name, topic):
    df_dataset = []

    with open(dataset_name, encoding='utf-8-sig') as FID:
        csvReader = csv.DictReader(FID, delimiter="\t")
        for key, row in enumerate(csvReader): 
            df_dataset.append({
                "label" : row['Topic'].strip().lower().replace('\n', ''),
                "text" : row['Full summary of comment']
            })

    df_dataset = pd.DataFrame(df_dataset)
    print(df_dataset['label'].value_counts(normalize=False))
    # exit(1)
    
    # Add in our classifier labels
    df_dataset['label'] = np.where(np.array(df_dataset['label']) == topic, 1, 0)
    df_dataset['label'].value_counts(normalize=False)

    # Train and test data
    class_len = len(df_dataset[df_dataset['label'] == 1])
    class_0_data = df_dataset[df_dataset.label.eq(0)].sample(class_len) 
    class_1_data = df_dataset[df_dataset.label.eq(1)].sample(class_len)
    train_test_data = pd.concat([class_0_data, class_1_data])  # 50/50 class split

    return train_test_data

# dataset_name = "/home/azureuser/cloudfiles/code/Users/Omololu.Makinde/Llama_tutorial/data/consultation2.csv"
# Topics = ["approach to the codes", "automated content moderation (user to user)", "governance and accountability"]

# for topic in Topics:
#     data = make_data(dataset_name, topic)
#     out_folder = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Input_Data"
#     out_file_path = Path(out_folder, topic+'.csv')
#     data.to_csv(out_file_path, index=False)

def kde_plot(data, label : str):
    if label == "1":
        c = "red"
    else:
        c = "blue"
    sns.kdeplot(np.array(data), bw_adjust=1, label=label, color=c, fill=True)

def NLI(premise, hypothesis, tokenizer, model):
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation=True)
    logits = model(input_ids)[0]
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_prob = probs[:, 1].item()
    return true_prob


def cos_sim(sentence, label, tokenizer, model):
    # Ensure label is a string for encoding
    label = [label]
    inputs = tokenizer.batch_encode_plus([sentence] + label, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    sentence_rep = outputs.last_hidden_state[0, :].mean(dim=0)
    label_rep = outputs.last_hidden_state[1, :].mean(dim=0)
    similarity = F.cosine_similarity(sentence_rep.unsqueeze(0), label_rep.unsqueeze(0))
    sim = similarity.item()
    return(sim + 1) / 2


infer_filepath = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Input_Data/approach to the codes.pdf"
s = time.time()

# Initiate logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/inference.log", level=logging.INFO, filemode="a")
logger.info("Script Started")

# Load models and tokenizer
tokenizer_nli = BartTokenizer.from_pretrained('joeddav/bart-large-mnli-yahoo-answers')
model_nli = BartForSequenceClassification.from_pretrained('joeddav/bart-large-mnli-yahoo-answers')
dataset = pd.read_csv("/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Input_Data/automated content moderation (user to user).csv")

topic = "governance and accountability"
c = 0
hypothesis = """
This text is about governance and accountability"""


# 1. make ravi's plot or box plot or bell curve distrubtion plot
# 2. determine threshold
# 3. go through edge cases that don't fit this
# dataset = """Hard to understand how a government entity can proscribe companies to comply, very technical, not enforceable - should just set goals and fines where required."""
# dataset = """Small businesses have limited resources and there should be guidelines on how the appeals process could be handled efficiently, with a clear and easy to follow criteria set on handling appeals to avoid disproportionate impact on resources."""  # 
# dataset = """Nothing further to add."""


### NLI ###
# 0
Prob = []
tp = 0
fn = 0
print(len(dataset))
sample_size = 400
for index, row in dataset.head(int(sample_size/2)).iterrows():
    premise = row['text']
    if pd.isna(premise):
        continue
    true_prob = NLI(premise, hypothesis, tokenizer_nli, model_nli)
    Prob.append(true_prob)
    # print(lindex, true_prob, "PRED:", round(true_prob), "LABEL:", row['label'])
    if round(true_prob) == row['label']:
        tp += 1
    else:
        fn += 1

# 1
Prob = []
tn = 0
fp = 0
for index, row in dataset.tail(int(sample_size/2)).iterrows():
    premise = row['text']
    if pd.isna(premise):
        continue
    true_prob = NLI(premise, hypothesis, tokenizer_nli, model_nli)
    Prob.append(true_prob)
    # print(index, true_prob, premise, "PRED:", round(true_prob), "LABEL:", row['label'])
    if round(true_prob) == row['label']:
        tn += 1
    else:
        fp += 1
print(np.array([[tp, fp], [fn, tn]]))


plt.legend()
# plt.savefig("/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/prob_curve2.png")
exit(1)
### COSINE SIMILARITY ###
tokenizer_cos = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_cos = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
dataset = """Hard to understand how a government entity can proscribe companies to comply, very technical, not enforceable - should just set goals and fines where required."""
dataset = """For a small business, some of the requirements for small multi risk businesses are "onerous and take significant time away from running and building a service. If the staff is 1-3 people, the same person will be responsible for all of these areas, alongside every other part of the business. There is some concern from ODDA members that high levels of governance requirements will have an impact on innovation in the dating and social discovery space, as many new services have extremely small staff and limited capacity.""" # 
dataset = """Nothing further to add."""

similarity = cos_sim(dataset, "governance and accountability", tokenizer_cos, model_cos)
print(similarity)
exit(1)
for index, row in dataset.iterrows():
    sentence = row['text']
    if pd.isna(sentence):
        continue
    similarity = cos_sim(sentence, "governance and accountability", tokenizer_cos, model_cos)
    if round(similarity) == row['label']:
        c += 1

    print(index, similarity, round(similarity), row['label'])

print(c, type(index), c / index)



print("run time:", time.time() - s, "s")
logger.info("Script Ended")
logger.info("run time: " + str(round(time.time() - s, 1)) + "s")