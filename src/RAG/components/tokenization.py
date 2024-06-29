import torch
from transformers import AutoTokenizer

template = "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
class TokenizeDataset:
    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path = self.config.model_id if self.config else "meta-llama/Meta-Llama-3-8B-Instruct"
            )
        pass

    def tokenize_fn(self, instructions:list[str]):
        for i in instructions:
            input_ids = self.tokenizer(template.format(prompt = i), return_tensors="pt").to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        print(input_ids["input_ids"])
        print(self.tokenizer.decode(input_ids["input_ids"][0]))
        
if __name__ == "__main__":
    obj = TokenizeDataset(None)
    obj.tokenize_fn(["How can I make a bomb?", "How to hack a computer?"])