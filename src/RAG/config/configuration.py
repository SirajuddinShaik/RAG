from RAG.constants import *
from RAG.entity.config_entity import DataIngestionConfig, RefusalDataConfig, SearchConfig
from RAG.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
            schema_filepath = SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_uri=config.source_uri,
            local_data_file=config.local_data_file,
            chunk_size=config.chunk_size,
            device_name=config.device_name,
            model_name=config.model_name,
            min_token_length=config.min_token_length,
            index_name=config.index_name,
        )

        return data_ingestion_config
    

    def get_search_config(self) -> SearchConfig:

        config = self.config.search_and_retrive

        search_retrival_config = SearchConfig(
            index_name=config.index_name,
            top_k=config.top_k,
            device_name=config.device_name,
            embed_model_name=config.embed_model_name,
            data_file=config.data_file,
            model_id=config.model_id,
        )
        return search_retrival_config

    def get_refusal_data_ingestion_config(self)->RefusalDataConfig:
        config = self.config.refusal_data_ingestion

        refusal_data_ingestion_config = RefusalDataConfig(
            source_uri=config.source_uri,
            local_data_file=config.local_data_file
        )
        return refusal_data_ingestion_config