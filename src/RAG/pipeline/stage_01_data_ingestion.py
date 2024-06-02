from RAG.components.data_ingestion import DataIngestion
from RAG.config.configuration import ConfigurationManager
from RAG import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeLine:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_data()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<")
        obj = DataIngestionPipeLine()
        obj.main()
        logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e