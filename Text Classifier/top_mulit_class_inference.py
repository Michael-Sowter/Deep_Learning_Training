# standard packages
import sys
import time
import os
from tika import parser
import json

# huggingface packages
from transformers import pipeline
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


def pred(topic, model_number, outfile):
# ------------------------------------------------------------------------------------------------------------------------------------------- #
    # Parse the pdf
    pdf_filepath = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Overview.pdf"
    parsed_file = parser.from_file(pdf_filepath)['content']

    # Split pdf into chunks
    embedding = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-small-Embedding-v0")  # get embeddings model
    text_splitter = SemanticChunker(embedding)  # apply the model to the type of split we want to perform (a semantic split)
    chunks = text_splitter.create_documents([parsed_file])
# ------------------------------------------------------------------------------------------------------------------------------------------- #
    # Load inferencing pipeline
    def inference_pipeline(model_path, max_length=512):
        pipe = pipeline("text-classification", model=model_path, max_length=max_length, truncation=True)
        return pipe

    # select best model
    best_model_path = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/Mod_" + str(model_number) + "/Best"
    infer = inference_pipeline(best_model_path)


    if not os.path.exists(outfile): 
        # If no existing file make a blank template
        res = {}
        for chunk in chunks:
            i = chunk.page_content
            res[i] = {}
    else:
        with open(str(outfile), 'r') as empt_par:
            res = json.load(empt_par)


    for chunk in chunks:
        i = chunk.page_content
        infer_res = infer(i.replace('\n\n', ''))[0]
        if infer_res['label'] == "Negative": 
            infer_res['score'] = 1 - infer_res['score']  # keep scoring between 0 and 1

        res[i][topic] = infer_res['score']

    with open(outfile, "w") as tagged_pars: 
        json.dump(res, tagged_pars, indent = 4)

    return


s = time.time()  # time script
model_number = 0
Topics = ["approach to the codes",                                 
"automated content moderation (user to user)",           
"governance and accountability"]

# If file exists, delete file (stops you from adding to existing file):
outfile = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Output_Data/Inf_res.json"
if os.path.exists(outfile):
    os.remove(outfile)

for topic in Topics:
    model_number+=1
    pred(topic, model_number, outfile)

print("run time:", time.time()-s, "s")