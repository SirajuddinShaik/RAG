import os
import re
import pandas as pd
import fitz
import requests
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
import os
from pinecone import Pinecone, ServerlessSpec

from RAG.entity.config_entity import DataIngestionConfig
from RAG import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig, api_key):
        self.config = config
        self.embedding_model = SentenceTransformer(
            model_name_or_path=self.config.model_name, device=self.config.device_name
        )
        self.pc = Pinecone(api_key)

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

    def text_formatter(self, text: str) -> str:
        cleaned_text = text.replace("\n", " ").strip()

        return cleaned_text

    def open_and_read_pdf(self, file_path=None) -> list:
        if not file_path:
            file_path = self.config.local_data_file
        docs = fitz.open(file_path)
        pages_and_text = []
        for page_no, page in tqdm(enumerate(docs)):
            text = page.get_text()
            text = self.text_formatter(text)
            pages_and_text.append(
                {
                    "page_number": page_no,
                    "page_char_count": len(text),
                    "page_word_count": len(text.split()),
                    "page_sentence_count": len(text.split(". ")),
                    "page_token_count": len(text) // 4,
                    "text": text,
                }
            )

        return pages_and_text

    def sentencize_and_chunk(self, pages_and_text: list):
        nlp = English()
        nlp.add_pipe("sentencizer")
        for item in tqdm(pages_and_text):
            item["sentences"] = list(nlp(item["text"]).sents)

            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            item["page_sentence_count_spacy"] = len(item["sentences"])

            item["sentence_chunks"] = self.chuck_data(item["sentences"])

            item["num_chunks"] = len(item["sentence_chunks"])
        return pages_and_text

    def chuck_data(self, input_list: list[str]) -> list[list[str]]:
        return [
            input_list[i : i + self.config.chunk_size]
            for i in range(0, len(input_list), self.config.chunk_size - 3)
        ]

    def split_chunks(self, pages_and_text: list, path=None) -> list:
        if not path:
            path = self.config.root_dir + "/data"
        pages_and_chunks = []
        for item in tqdm(pages_and_text):
            page_no = item["page_number"]
            for i, sentence_chunk in enumerate(item["sentence_chunks"]):
                chunk_dict = {}
                joined_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_chunk = re.sub(r"\.([A-Z])", r". \1", joined_chunk)
                chunk_dict["index"] = f"{page_no+1}_{i}"
                chunk_dict["sentence_chunk"] = joined_chunk
                chunk_dict["chunk_char_count"] = len(joined_chunk)
                chunk_dict["chunk_word_count"] = len(joined_chunk.split())
                chunk_dict["chunk_token_count"] = len(joined_chunk) // 4

                pages_and_chunks.append(chunk_dict)
                pd.DataFrame(pages_and_chunks).drop(
                    columns=[
                        "chunk_token_count",
                        "chunk_word_count",
                        "chunk_char_count",
                    ]
                ).to_csv(f"{path}.csv")
        return pages_and_chunks

    def embed_chunks(self, pages_and_chunks: list) -> list:
        self.embedding_model.to(self.config.device_name)
        logger.info(">>>>>>>> Embedding Started <<<<<<<<<")
        df = pd.DataFrame(pages_and_chunks)
        pages_and_chunks_over_min_token_len = df[
            df["chunk_token_count"] > self.config.min_token_length
        ]
        text_chunks = [
            item["sentence_chunk"]
            for item in pages_and_chunks_over_min_token_len.to_dict(orient="records")
        ]
        text_chunks_embeddings = self.embedding_model.encode(
            text_chunks,
            batch_size=32,
            convert_to_tensor=True,
        )

        temp = text_chunks_embeddings.cpu()
        del text_chunks_embeddings
        text_chunks_embeddings = temp
        logger.info(">>>>>>>> Embedding Completed <<<<<<<<<")
        # for i, data in enumerate(pages_and_chunks_over_min_token_len):
        pages_and_chunks_over_min_token_len["embeddings"] = [
            i for i in text_chunks_embeddings
        ]
        return pages_and_chunks_over_min_token_len.to_dict(orient="records")

    def to_vector_database(self, pages_and_chunks_over_min_token_len, index_name=None):
        if not index_name:
            index_name = self.config.index_name
        if index_name in self.pc.list_indexes().names():
            logger.warning(f"Index Already Exist! Deleting Index:{index_name}")
            self.pc.delete_index(index_name)
        self.pc.create_index(
            name=index_name,
            dimension=pages_and_chunks_over_min_token_len[0]["embeddings"].shape[0],
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Data is Going To Load to Pinecone!")
        index = self.pc.Index(index_name)
        vectors = [
            {"id": item["index"], "values": item["embeddings"]}
            for item in pages_and_chunks_over_min_token_len
        ]
        vectors = [vectors[i : i + 32] for i in range(0, len(vectors), 32)]
        for batch in tqdm(vectors):
            index.upsert(vectors=batch)
        logger.info("Vectors Loaded to Pinecone")
