import os
import re
import requests
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone

from RAG.entity.config_entity import SearchConfig
from RAG.utils.common import setup_env
from RAG.utils.prompts import PROMPTS, PROMPTS2, PROMPTS3


class SearchAndAnswer:
    def __init__(self, config: SearchConfig, pc_key) -> None:
        self.config = config
        self.pc = Pinecone(api_key=pc_key)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=self.config.embed_model_name,
            device=self.config.device_name,
        )
        self.dfs = {
            "slot-1": "",
            "slot-2": "",
            "slot-3": "",
            "slot-4": "",
        }
        self.paths = {
            "slot-1": "",
            "slot-2": "",
            "slot-3": "",
            "slot-4": "",
        }
        if self.config.device_name == "cuda":
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.config.model_id
            )
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers.utils import is_flash_attn_2_available

            # 1. Create quantization config for smaller model loading (optional)
            # Requires !pip install bitsandbytes accelerate, see: https://github.com/TimDettmers/bitsandbytes, https://huggingface.co/docs/accelerate/
            # For models that require 4-bit quantization (use this if you have low GPU memory available)
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16
            )

            if (is_flash_attn_2_available()) and (
                torch.cuda.get_device_capability(0)[0] >= 8
            ):
                attn_implementation = "flash_attention_2"
            else:
                attn_implementation = "sdpa"
            print(f"[INFO] Using attention implementation: {attn_implementation}")

            model_id = "meta-llama/Meta-Llama-3-8B"  # (we already set this above)
            print(f"[INFO] Using model_id: {model_id}")

            # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.config.model_id
            )

            # 4. Instantiate the model
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.model_id,
                torch_dtype=torch.float16,  # datatype to use, we want float16
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,  # use full memory
                attn_implementation=attn_implementation,
            )
        else:
            pass

    def setup_pd(self, paths):
        for slot in paths:
            path = paths[slot]
            if path != "":
                self.dfs[slot] = pd.read_csv(f"{path}.csv")
                self.dfs[slot].set_index("index", inplace=True)
        self.paths = paths

    def retrive_similar_enbeddings(self, query: str, index=None):
        if not index:
            index = self.config.index_name
        index = self.pc.Index(self.config.index_name)
        embeddings = self.embedding_model.encode(query)
        query_results = index.query(
            vector=embeddings.tolist(), top_k=self.config.top_k, include_values=True
        )
        return query_results

    def fetch_chunks(self, query_results, index):
        query_results = query_results["matches"]
        for item in query_results:
            item["sentence_chunk"] = self.dfs[index].loc[item["id"]]["sentence_chunk"]
        return query_results

    def prompt_formatter(
        self, query: str, context_items: list[dict], base_prompt: str
    ) -> str:
        """
        Augments query with text-based context from context_items.
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join(
            [
                "Index(" + item["id"] + ")-" + item["sentence_chunk"]
                for item in context_items
            ]
        )

        # Update base prompt with context items and query
        base_prompt = base_prompt.format(context=context, query=query)

        return base_prompt

    def ask_gpu(
        self,
        base_prompt,
        temperature,
        max_new_tokens=512,
        format_answer_text=True,
        return_answer_only=True,
    ):

        # Create prompt template for instruction-tuned model
        dialogue_template = [{"role": "user", "content": base_prompt}]
        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation=dialogue_template, tokenize=False, add_generation_prompt=True
        )
        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(
            self.config.device_name
        )

        # Generate an output of tokens
        outputs = self.llm_model.generate(
            **input_ids,
            temperature=temperature,
            do_sample=True,
            max_new_tokens=max_new_tokens,
        )

        # Turn the output tokens into text
        output_text = self.tokenizer.decode(outputs[0])

        if format_answer_text:
            # Replace special tokens and unnecessary help message
            output_text = (
                output_text.replace(prompt, "")
                .replace("<bos>", "")
                .replace("<eos>", "")
                .replace("Sure, here is the answer to the user query:\n\n", "")
            )

        # Only return the answer without the context items
        if return_answer_only:
            return output_text

        return

    def ask_cpu(self, base_prompt, temperature, hf_key, model, max_new_tokens=512):
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        API_TOKEN = hf_key
        payload = {
            "inputs": base_prompt,
            "parameters": {
                "temperature": temperature,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None

    def ask(self, query, context_items, prompt_type, hf_key, model):
        prompt_set = PROMPTS3[prompt_type]
        context = "- " + "\n- ".join(
            [
                "Index[" + item["id"] + "]-" + item["sentence_chunk"]
                for item in context_items
            ]
        )

        # Update base prompt with context items and query
        base_prompt = prompt_set["prompt"].format(context=context, query=query)
        base_prompt = self.clean_prompt(base_prompt)
        if self.config.device_name == "cuda":
            answer = self.ask_gpu(base_prompt, prompt_set["temperature"])
        else:
            answer = self.ask_cpu(base_prompt, prompt_set["temperature"], hf_key, model)
        return answer

    def clean_prompt(self, prompt):
        # Remove leading and trailing spaces
        # prompt = prompt.strip()

        # Replace multiple spaces with a single space
        prompt = re.sub(r"\s+", " ", prompt)

        # Remove all special characters except letters, digits, and basic punctuation
        # Allowed characters: a-z, A-Z, 0-9, space, and basic punctuation (.,!?')
        allowed_characters = re.compile(r'^a-zA-Z0-9\s.,!?\'"-')

        # Remove characters that are not in the allowed set
        cleaned_prompt = allowed_characters.sub("", prompt)
        print(cleaned_prompt)
        return cleaned_prompt
