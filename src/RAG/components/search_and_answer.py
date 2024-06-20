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
from RAG.utils.prompts import PROMPTS, PROMPTS2, PROMPTS3, LLAMA


class SearchAndAnswer:
    def __init__(self, config: SearchConfig, pc_key, llm_model = None) -> None:
        self.config = config
        self.pc = Pinecone(api_key=pc_key)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=self.config.embed_model_name,
            device=self.config.device_name,
        )
        self.dfs = {
            "slot-1": pd.DataFrame(),
            "slot-2": pd.DataFrame(),
            "slot-3": pd.DataFrame(),
            "slot-4": pd.DataFrame(),
        }
        self.paths = {
            "slot-1": "",
            "slot-2": "",
            "slot-3": "",
            "slot-4": "",
        }
        if self.config.device_name == "cuda":
            self.chat = LLAMA["system"].format(
                system_msg="You are a helpful AI assistant"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.config.model_id
            )
            self.llm_model = llm_model
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
        index = self.pc.Index(index)
        embeddings = self.embedding_model.encode(query)
        query_results = index.query(
            vector=embeddings.tolist(), top_k=self.config.top_k, include_values=True
        )
        print(query_results)
        return query_results

    def fetch_chunks(self, query_results, index):
        query_results = query_results["matches"]
        print(index)
        if self.paths[index] != "":
            for item in query_results:
                item["sentence_chunk"] = self.dfs[index].loc[item["id"]][
                    "sentence_chunk"
                ]
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
        format_answer_text=False,
        return_answer_only=True,
    ):

        # Create prompt template for instruction-tuned model
        # dialogue_template = [{"role": "user", "content": base_prompt}]
        # # Apply the chat template
        # prompt = self.tokenizer.apply_chat_template(
        #     conversation=dialogue_template, tokenize=False, add_generation_prompt=True
        # )
        # Tokenize the prompt
        input_ids = self.tokenizer(base_prompt, return_tensors="pt").to(
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
        self.chat.append(output_text)
        if format_answer_text:
            # Replace special tokens and unnecessary help message
            output_text = (
                output_text.replace(base_prompt, "")
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

    def ask(self, query, context_items, prompt_type, hf_key, model, index):
        if (
            index == "chat"
            and self.config.device_name == "cuda"
            and model == "Llama-3(gpu)"
        ):
            if prompt_type == "system":
                self.chat = ""
            if prompt_type not in ["chat", "system"]:
                return "currently chat or system is suported for llama 3 gpu"
            prompt = LLAMA[prompt_type].format(msg=query)
            self.chat += prompt
            prompt_set = {"prompt": self.chat, "temperature": 1}
            answer = self.ask_gpu(base_prompt, prompt_set["temperature"])
            return answer
        elif prompt_type != "system" and model != "Llama-3(gpu)":
            if prompt_type == "system":
                return "system message cant be modified!"
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
            if model == "Llama-3(gpu)" and self.config.device_name == "cuda":
                base_prompt = (
                    LLAMA["system"].format(msg="You are a helpful AI assistant")
                    + "\n\n"
                    + LLAMA["user"].format(msg=base_prompt)
                )
        if self.config.device_name == "cuda" and model == "Llama-3(gpu)":
            answer = self.ask_gpu(base_prompt, prompt_set["temperature"])
        elif model != "Llama-3(gpu)":
            answer = self.ask_cpu(base_prompt, prompt_set["temperature"], hf_key, model)
        else:
            answer = self.err_msg(
                "Your Instruction setting is Wrong! eg: selecting gpu model without gpu"
            )
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

    def err_msg(self, msg):
        return [{"generated_text": msg}]
