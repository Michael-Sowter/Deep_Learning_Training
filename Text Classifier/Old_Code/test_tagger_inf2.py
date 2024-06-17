import logging
import time
import os
from src.make_models.topic_tagger import Inference

# Initiate logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/inference.log", level=logging.INFO, filemode="w")
logger.info("Script Started")

s = time.time()
# Topics = ["approach to the codes", "automated content moderation (user to user)", "governance and accountability"]
Topics = ["approach to the codes", "register of risks", "automated content moderation (user to user)", "governance and accountability", "icjg", "user reporting and complaints (u2u and search)", """service’s risk assessment""", "content moderation (user to user)", "user access to services (u2u)", "enhanced user control (u2u)"]# ["approach to the codes", "automated content moderation (user to user)", "governance and accountability"]
pdf_filepath = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Input_Data/governance and accountability.pdf"
out_filepath = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Output_Data/dl_sup_new.json"
embeddings_model_name = "avsolatorio/GIST-small-Embedding-v0"

# If file exists, delete file (stops you from adding to existing file):
if os.path.exists(out_filepath):
    os.remove(out_filepath)

topic="governance and accountability"
logger.info(f"Topic: {topic}")
best_model_path = f"/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/{topic}/Mod_2/lr:1e-05_wd:0.01/Best"
tci = Inference(topic, logger, pdf_filepath, out_filepath, embeddings_model_name, best_model_path)
parsed_file = tci.parse_file()
chunks = tci.pdf_splitter(parsed_file)
pipe = tci.load_inference_pipe()
res = tci.create_or_load_json(chunks)
tci.append_inference_result(chunks, pipe, res)



print("run time:", time.time()-s, "s")
logger.info("Script Ended")
logger.info("run time: " + str(round(time.time()-s, 1)) + "s")