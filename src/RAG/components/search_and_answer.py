import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pinecone import Pinecone

from RAG.entity.config_entity import SearchConfig
from RAG.utils.common import setup_env

class SearchAndAnswer:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        setup_env()
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index(self.config.index_name)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=self.config.embed_model_name, 
            device=self.config.device_name
        )
        self.df = pd.read_csv(self.config.data_file)
        self.df.set_index("index",inplace=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.config.model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.config.model_id, 
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 low_cpu_mem_usage=False, # use full memory 
                                                 attn_implementation="sdpa") # which attention version to use

        self.llm_model.to("cuda")


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


    def prompt_formatter(self, query: str, context_items: list[dict]) -> str:
        """
        Augments query with text-based context from context_items.
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join(["Index-"+item["id"]+":"+item["sentence_chunk"] for item in context_items])

        # Create a base prompt with examples to help the model
        # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
        # We could also write this in a txt file and import it in if we wanted.
        base_prompt = """Based on the following context items, please answer the query.
        Give yourself room to think by extracting relevant passages from the context before answering the query.
        Don't return the thinking, only return the answer.
        Make sure your answers are as explanatory as possible.
        Use the following examples as reference for the ideal answer style.
        \nExample 1:
        Query: What are the fat-soluble vitamins?
        Answer: Index-9_1: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
        \nExample 2:
        Query: What are the causes of type 2 diabetes?
        Answer: Index-52_0: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
        \nExample 3:
        Query: What is the importance of hydration for physical performance?
        Answer: Index-109_3: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
        \nNow use the following context items to answer the user query:
        {context}
        \nRelevant passages: <extract relevant passages from the context here with index which is more relevent>
        User query: {query}
        Answer:"""

        # Update base prompt with context items and query   
        base_prompt = base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": base_prompt}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False,
                                            add_generation_prompt=True)
        return prompt
    
    def ask(self,
            query,
            context_items,
            temperature=0.7,
            max_new_tokens=512,
            format_answer_text=True, 
            return_answer_only=True,):
        
        prompt = self.prompt_formatter(query=query,
                                context_items=context_items)
        
        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate an output of tokens
        outputs = self.llm_model.generate(**input_ids,
                                    temperature=temperature,
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
        
        return output_text