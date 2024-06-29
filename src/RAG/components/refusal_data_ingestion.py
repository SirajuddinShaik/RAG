import io
import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from RAG import logger

class RefusalDataIngestion:
    def __init__(self, config):
        self.config = config

    def get_harmful_instructions(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info("File Doesn't Exist!>>> let's Download")
            response = requests.get(self.config.source_uri)

            dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            dataset.to_csv(self.config.local_data_file)
            logger.info(f">>>>>> Downloaded to {self.config.local_data_file}")
        else:
            logger.info("File Already Exist!")
            dataset = pd.read_csv(self.config.local_data_file)
        instructions = dataset['goal'].tolist()
        train, test = train_test_split(instructions, test_size=0.2, random_state=42)
        return train, test
    
    def get_harmless_instructions(self):
        hf_path = 'tatsu-lab/alpaca'
        dataset = load_dataset(hf_path)

        # filter for instructions that do not have inputs
        instructions = []
        for i in range(len(dataset['train'])):
            if dataset['train'][i]['input'].strip() == '':
                instructions.append(dataset['train'][i]['instruction'])

        train, test = train_test_split(instructions, test_size=0.2, random_state=42)
        return train, test