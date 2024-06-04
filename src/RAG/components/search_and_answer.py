import os
import requests
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone

from RAG.entity.config_entity import SearchConfig
from RAG.utils.common import setup_env

class SearchAndAnswer:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index(self.config.index_name)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=self.config.embed_model_name, 
            device=self.config.device_name
        )
        self.df = pd.read_csv(self.config.data_file)
        self.df.set_index("index",inplace=True)
        if self.config.device_name == "cuda":
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.config.model_id)
            self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.config.model_id, 
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 low_cpu_mem_usage=False, # use full memory 
                                                 attn_implementation="sdpa") # which attention version to use

            self.llm_model.to("cuda")
        else:
            pass


    def retrive_similar_enbeddings(self, query: str):
        embeddings = self.embedding_model.encode(query)
        query_results = self.index.query(
            vector=embeddings.tolist(),
            top_k=self.config.top_k,
            include_values=True
        )
        return query_results
    
    def fetch_chunks(self, query_results):
        query_results = query_results["matches"]
        for item in query_results:
            item["sentence_chunk"] = self.df.loc[item["id"]]["sentence_chunk"]
        return query_results


    def prompt_formatter(self, query: str, context_items: list[dict], base_prompt: str) -> str:
        """
        Augments query with text-based context from context_items.
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join(["Index("+item["id"]+")-"+item["sentence_chunk"] for item in context_items])
        
        # Update base prompt with context items and query   
        base_prompt = base_prompt.format(context=context, query=query)

        return base_prompt
    
    def ask_gpu(self,
            query,
            context_items,
            prompt_template,
            max_new_tokens=512,
            format_answer_text=True, 
            return_answer_only=True,):
        
        base_prompt = self.prompt_formatter(query, context_items, prompt_template)
        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": base_prompt}
        ]
        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False,
                                            add_generation_prompt=True)
        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.config.device_name)

        # Generate an output of tokens
        outputs = self.llm_model.generate(**input_ids,
                                    temperature=prompt_template["temperature"],
                                    do_sample=True,
                                    max_new_tokens=max_new_tokens)
        
        # Turn the output tokens into text
        output_text = self.tokenizer.decode(outputs[0])

        if format_answer_text:
            # Replace special tokens and unnecessary help message
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

        # Only return the answer without the context items
        if return_answer_only:
            return output_text
        
        return 
    
    def ask_cpu(self,
            query,
            context_items,
            prompt_template,
            max_new_tokens=512):
        prompt = self.prompt_formatter(query, context_items, prompt_template["prompt"])
        print(prompt)
        API_URL = f"https://api-inference.huggingface.co/models/{self.config.model_id}"
        API_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": prompt_template["temperature"],
                "max_new_tokens": max_new_tokens,
                "return_full_text": False
            }
        }
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None


    def ask(self, query, context_items, prompt_template):
        if self.config.device_name == "cuda":
            answer = self.ask_gpu(query, context_items, prompt_template)
        else:
            answer = self.ask_cpu(query, context_items, prompt_template)
        return answer