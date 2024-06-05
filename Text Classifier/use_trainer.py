import trainer_class
import logging
import sys
import time

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
    learning_rate = 50
    weight_decay = 0.01
    logger.info(f"Training new model... ")
    logger.info(f"PARAMETERS \t Learning Rate: {learning_rate} \t Weight Decay: {weight_decay} \t Epochs: {epochs}")
    logger.info("Script Ended")
    model_output_path = "/home/azureuser/cloudfiles/code/Users/Michael.Sowter/Deep_Learning_Training/Text Classifier/Models/"+ topic + "/Batch/lr:"+str(learning_rate) + "_wd:" + str(weight_decay)
    tct = trainer_class.TextClassifierTrainer(dataset_name, topic, learning_rate, epochs, pretrained_model_name, weight_decay, model_output_path)
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
#     tct = trainer_class.TextClassifierTrainer(dataset_name, topic, learning_rate, epochs, pretrained_model_name, weight_decay)
#     dataset, eval_data = tct.create_dataset()
#     trainer = tct.train_model(dataset)
        
#     # Train model
#     print(topic)

#     # Get best model scores
#     print(trainer.evaluate())

#     break  # stop at first topic

# print("run time:", time.time()-s, "s")