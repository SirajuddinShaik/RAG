import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig


def get_model():

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    print(f"[INFO] Using attention implementation: {attn_implementation}")

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # (we already set this above)
    print(f"[INFO] Using model_id: {model_id}")

    # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    # 4. Instantiate the model
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,  # datatype to use, we want float16
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,  # use full memory
        attn_implementation=attn_implementation,
    ) 
    return llm_model