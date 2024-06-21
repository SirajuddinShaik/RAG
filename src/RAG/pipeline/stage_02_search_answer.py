from RAG.components.search_and_answer import SearchAndAnswer
from RAG.config.configuration import ConfigurationManager
from RAG import logger
from RAG.utils.prompts import PROMPTS
from RAG.utils.common import setup_env

STAGE_NAME = "Search Answer stage"


class SearchAnswerPipeline:
    def __init__(self, pc_key, llm_model=None) -> None:
        setup_env()
        config = ConfigurationManager()
        self.search_config = config.get_search_config()
        self.query_answer = SearchAndAnswer(self.search_config, pc_key, llm_model)

    def main(self, query: str, prompt: str):
        query_embeddings = self.query_answer.retrive_similar_enbeddings(query)
        query_results = self.query_answer.fetch_chunks(query_embeddings)
        output_text = self.query_answer.ask(query, query_results, prompt)
        return output_text

    def chainlit_prompt(self, query: str, prompt: str, pc_key, hf_key, index, model):
        try:
            if index != "chat":
                query_embeddings = self.query_answer.retrive_similar_enbeddings(
                    query, index
                )
                query_results = self.query_answer.fetch_chunks(query_embeddings, index)
            else:
                query_results = []
            output_text = self.query_answer.ask(
                query, query_results, prompt, hf_key, model, index
            )
            print(output_text)
            return output_text
        except Exception as e:
            print(e)
            return "data or pdf not loaded in the slot or " + str(e)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<")
        obj = SearchAnswerPipeline("Pc-key...")
        query = ""
        prompt = "detailed_prompt"
        obj.query_answer.setup_pd(
            {
                "slot-1": obj.search_config.data_file,
                "rag-index": obj.search_config.data_file,
                "slot-3": "",
                "slot-4": "",
                "slot-5": "",
            }
        )
        obj.chainlit_prompt(
            "how many bones does our body has?",
            prompt,
            "",
            "",
            "rag-index",
            "google/gemma-2b-it",
        )
        # while query != ".exit":
        #     query = input("Enter The Query: ")
        #     output_text = obj.main(query, prompt)
        #     print(output_text)
        # logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
