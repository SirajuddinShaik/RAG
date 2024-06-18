import os
import random
from RAG.components.data_ingestion import DataIngestion
from RAG.config.configuration import ConfigurationManager
from RAG import logger
from RAG.utils.common import setup_env

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeLine:
    def __init__(self) -> None:
        setup_env()
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()

    def main(self):
        data_ingestion = DataIngestion(self.data_ingestion_config, os.environ["PINECONE_API_KEY"])
        data_ingestion.download_data()
        pages_and_text = data_ingestion.open_and_read_pdf()
        pages_and_text = data_ingestion.sentencize_and_chunk(pages_and_text)
        pages_and_chunks = data_ingestion.split_chunks(pages_and_text)
        pages_and_chunks_over_min_token_len = data_ingestion.embed_chunks(pages_and_chunks)
        # data_ingestion.to_vector_database(pages_and_chunks_over_min_token_len)
        return pages_and_chunks_over_min_token_len

    def load_to_pincone(self, api_key, file_path, index_name):
        data_ingestion = DataIngestion(self.data_ingestion_config, api_key)
        pages_and_text = data_ingestion.open_and_read_pdf(file_path)
        pages_and_text = data_ingestion.sentencize_and_chunk(pages_and_text)
        pages_and_chunks = data_ingestion.split_chunks(pages_and_text, file_path)
        pages_and_chunks_over_min_token_len = data_ingestion.embed_chunks(pages_and_chunks)
        data_ingestion.to_vector_database(pages_and_chunks_over_min_token_len, index_name)


    def store_tokens(self, api_key, file_path, index_name):
        data_ingestion = DataIngestion(self.data_ingestion_config, api_key)
        pages_and_text = data_ingestion.open_and_read_pdf(file_path)
        pages_and_text = data_ingestion.sentencize_and_chunk(pages_and_text)
        pages_and_chunks = data_ingestion.split_chunks(pages_and_text, file_path)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<")
        obj = DataIngestionPipeLine()
        pages_and_text = obj.main()
        # print(random.sample(pages_and_text, k=1))
        logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e