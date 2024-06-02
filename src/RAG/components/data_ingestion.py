import os

import requests
from RAG.entity.config_entity import DataIngestionConfig
from RAG import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info("File Doesn't Exist!>>> let's Download")
            response = requests.get(self.config.source_uri)

            if response.status_code == 200:
                f = open(self.config.local_data_file, "wb")
                f.write(response.content)
                logger.info(f">>>>>> Downloaded to {self.config.local_data_file}")
            else:
                logger.info(f"Download Error, Code:{response.status_code}")
        else:
            logger.info("File Already Exist!")
    
