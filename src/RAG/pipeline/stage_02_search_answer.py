from huggingface_hub import login
import os

from RAG.components.search_and_answer import SearchAndAnswer
from RAG.config.configuration import ConfigurationManager
from RAG import logger
from RAG.utils.common import setup_env

STAGE_NAME = "Search Answer stage"

class SearchAnswerPipeline:
    def __init__(self) -> None:
        login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
        setup_env()
        config = ConfigurationManager()
        search_config = config.get_search_config()
        self.query_answer = SearchAndAnswer(search_config)

    def main(self, query:str):
        query_embeddings = self.query_answer.retrive_similar_enbeddings(query)
        query_results = self.query_answer.fetch_chunks(query_embeddings)
        output_text = self.query_answer.ask(query, query_results)
        return output_text
    
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<")
        obj = SearchAnswerPipeline()
        query = input("Enter The Query: ")
        output_text = obj.main(query=query)
        print(output_text)
        logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
