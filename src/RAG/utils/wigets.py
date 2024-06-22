from chainlit.input_widget import Select, TextInput

model_select = Select(
    id="Model",
    label="select - Model",
    values=[
        "openai-community/gpt2",
        "Llama-3(gpu)",
        "google/gemma-2b-it",
        "google/gemma-2b",
        "openai-community/gpt2-xl",
        "microsoft/Phi-3-vision-128k-instruct",
    ],
    initial_index=0,
)

query_type = Select(
    id="Query",
    label="Select Task - Type",
    values=[
        "chat",
        "system",
        "detailed_prompt",
        "short_prompt",
        "summary_prompt",
        "explanation_prompt",
        "opinion_prompt",
        "instruction_prompt",
    ],
    initial_index=0,
)
pdf_select = Select(
    id="Pdf",
    label="Select Pdf - Slot",
    values=["chat", "slot-1", "slot-2", "slot-3", "nutrition"],
    initial_index=0,
)
hf_api = TextInput(
    id="hf_key",
    label="Enter your Huggingface API key",
    placeholder="hf-...",
)
pc_api = TextInput(
    id="pc_key", label="Enter your Pinecone API key", placeholder="sk-..."
)
