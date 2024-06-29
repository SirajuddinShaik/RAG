from RAG.components.refusal_data_ingestion import RefusalDataIngestion
from RAG.config.configuration import ConfigurationManager
from RAG import logger

STAGE_NAME = "Refusal Data Ingestion stage"

class RefugalDataIngestionPipeline:
    def __init__(self) -> None:
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_refusal_data_ingestion_config()

    def main(self):
        data_ingestion = RefusalDataIngestion(self.data_ingestion_config)

        data_ingestion.get_harmful_instructions()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<")
        obj = RefugalDataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
