import os
import random
from RAG.components.internet_scrape import InternetScraper
from RAG.config.configuration import ConfigurationManager
from RAG import logger
from RAG.utils.common import setup_env
from RAG.utils.prompts import INTERNET_RETRIVE_PROMPT

STAGE_NAME = "Internet Query stage"

class InternetQueryPipeLine:
    def __init__(self) -> None:
        setup_env()
        config = ConfigurationManager()
        self.internet_query_config = config.get_internet_query_config()

    def main(self,query):
        internet_query = InternetScraper(self.internet_query_config)
        urls = internet_query.get_search_results(query=query)
        scrape_data = internet_query.scrape_urls(urls)
        summerized_content = internet_query.summarize_content([data['content'] for data in scrape_data],query=query)
        prompt = INTERNET_RETRIVE_PROMPT["internet"].format(context_from_internet =summerized_content,user_question=query )
        return prompt

    


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<")
        obj = InternetQueryPipeLine()
        prompt = obj.main("What is Python?")
        print(prompt)
        # print(random.sample(pages_and_text, k=1))
        logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e