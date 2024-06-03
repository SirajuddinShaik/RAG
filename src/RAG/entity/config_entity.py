from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_uri: str
    local_data_file: Path
    chunk_size: int
    device_name: str
    model_name: str
    min_token_length: int
    index_name: str

@dataclass(frozen=True)
class SearchConfig:
    index_name: str
    top: int
