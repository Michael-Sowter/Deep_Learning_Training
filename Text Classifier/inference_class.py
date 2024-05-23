# standard packages
import sys
import time
import os
from tika import parser
import json
import logging

# huggingface packages
from transformers import pipeline
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings



class TextClassiferInference:
    def __init__(self, logger, pdf_filepath : str, out_filepath : str, embeddings_model_name : str, tuned_model_name : str):
        self.logger = logger
        self.pdf_filepath = pdf_filepath
        self.out_filepath = out_filepath
        self.embeddings_model_name = embeddings_model_name
        self.tuned_model_name = tuned_model_name

    def parse_file(self):
        self.logger.info(f"Parsing PDF file: {self.pdf_filepath}")
        parsed_file = parser.from_file(self.pdf_filepath)['content']
        return parsed_file
    
    def pdf_splitter(self, parsed_file):
        self.logger.info("Split text into chunks using semantic chunker")
        embedding = HuggingFaceEmbeddings(model_name=embeddings_model_name)  # get embeddings model
        text_splitter = SemanticChunker(embedding)  # apply the model to the type of split we want to perform (a semantic split)
        chunks = text_splitter.create_documents([parsed_file])
        return chunks

    def load_inference_pipe(self):
        self.logger.info("Calibrate text inferencing pipeline")
        pipe = pipeline("text-classification", model=self.tuned_model_name, max_length=512, truncation=True)
        return pipe
    
    def create_or_load_json(self, chunks):
        if not os.path.exists(self.out_filepath): 
            self.logger.info("Output json file NOT found ...")
            res = {}
            for chunk in chunks:
                i = chunk.page_content
                res[i] = {}
                self.logger.info("Results template created")
        else:
            self.logger.info("Output json file FOUND ...")
            with open(str(self.out_filepath), 'r') as empt_par:
                res = json.load(empt_par)
                self.logger.info("Outfile loaded")
        return res

    def append_inference_result(self, chunks, infer, res):
        self.logger.info("Append inferencing result to json file")
        for chunk in chunks:
            i = chunk.page_content
            infer_res = infer(i.replace('\n\n', ''))[0]
            if infer_res['label'] == "Negative": 
                infer_res['score'] = 1 - infer_res['score']  # keep scoring between 0 and 1
            res[i][topic] = infer_res['score']

        with open(self.out_filepath, "w") as tagged_pars: 
            json.dump(res, tagged_pars, indent = 4)
            self.logger.info("Results saved to json")
        

if __name__ == "__main__":

    # Initiate logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="inference.log", level=logging.INFO, filemode="w")
    logger.info("Script Started")

    s = time.time()
    Topics = ["approach to the codes", "automated content moderation (user to user)", "governance and accountability"]
    pdf_filepath = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Input_Data/Overview.pdf"
    out_filepath = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Output_Data/Inf_res.json"
    embeddings_model_name = "avsolatorio/GIST-small-Embedding-v0"
    
    # If file exists, delete file (stops you from adding to existing file):
    if os.path.exists(out_filepath):
        os.remove(out_filepath)

    for topic in Topics:
        logger.info(f"Topic: {topic}")
        best_model_path = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/Mod_" + topic + "/Best"
        tci = TextClassiferInference(logger, pdf_filepath, out_filepath, embeddings_model_name, best_model_path)
        parsed_file = tci.parse_file()
        chunks = tci.pdf_splitter(parsed_file)
        pipe = tci.load_inference_pipe()
        res = tci.create_or_load_json(chunks)
        tci.append_inference_result(chunks, pipe, res)


print("run time:", time.time()-s, "s")
logger.info("Script Ended")
logger.info("run time: " + str(round(time.time()-s, 1)) + "s")