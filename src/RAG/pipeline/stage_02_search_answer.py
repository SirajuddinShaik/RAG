from huggingface_hub import login
import os

from RAG.components.search_and_answer import SearchAndAnswer
from RAG.config.configuration import ConfigurationManager
from RAG import logger
from RAG.utils.prompts import PROMPTS
from RAG.utils.common import setup_env

STAGE_NAME = "Search Answer stage"

class SearchAnswerPipeline:
    def __init__(self,pc_key) -> None:
        setup_env()
        config = ConfigurationManager()
        self.search_config = config.get_search_config()
        self.query_answer = SearchAndAnswer(self.search_config,pc_key)

    def main(self, query: str, prompt: str):
        query_embeddings = self.query_answer.retrive_similar_enbeddings(query)
        query_results = self.query_answer.fetch_chunks(query_embeddings)
        output_text = self.query_answer.ask(query, query_results, prompt)
        return output_text
    
    def chainlit_prompt(self, query: str, prompt: str, pc_key, hf_key, index, model):
        query_embeddings = self.query_answer.retrive_similar_enbeddings(query, pc_key)
        query_results = self.query_answer.fetch_chunks(query_embeddings, index)
        output_text = self.query_answer.ask(query, query_results, prompt, hf_key, model)
        print(output_text)
        return output_text
    
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<")
        obj = SearchAnswerPipeline("650ac970-f431-45fb-8a42-f9c53207a7c2")
        query = ""
        prompt = "detailed_prompt"
        obj.query_answer.setup_pd({"slot-1": obj.search_config.data_file, "slot-2": "", "slot-3": "", "slot-4": "", "slot-5": ""})
        obj.chainlit_prompt("what are skills", prompt, "650ac970-f431-45fb-8a42-f9c53207a7c2", "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","slot-1", "google/gemma-2b-it")
        # while query != ".exit":
        #     query = input("Enter The Query: ")
        #     output_text = obj.main(query, prompt)
        #     print(output_text)
        # logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
