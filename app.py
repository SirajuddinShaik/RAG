from io import BytesIO
from RAG.components.model import get_model
import fitz
from pinecone import Pinecone

import chainlit as cl
from tqdm.auto import tqdm

from RAG.pipeline.stage_01_data_ingestion import DataIngestionPipeLine
from RAG.pipeline.internet_query import InternetQueryPipeLine
from RAG.pipeline.stage_02_search_answer import SearchAnswerPipeline
from RAG.utils.wigets import *

llm_model = None
import torch

if torch.cuda.is_available(): # which attention version to use
    llm_model ,tokenizer= get_model()

# internet Query
internet_query_obj = InternetQueryPipeLine()


@cl.on_chat_start
async def start():
    # Initialize with empty values
    # cl.user_session.set("user", {"pc":"","hf":"","key_status":False})

    cl.user_session.set("pc", "")
    cl.user_session.set("hf", "")
    cl.user_session.set("model", "")
    cl.user_session.set("task", "")
    cl.user_session.set("paths", {"slot-1": "", "slot-2": "", "slot-3": ""})
    cl.user_session.set("slot", "chat")
    cl.user_session.set("err", "Setup Api Keys")
    cl.user_session.set("api_status", False)
    msg = cl.Message(content=f"Setup Your Api keys and Start")
    await msg.send()
    # Create settings with the updated select elements
    settings = await cl.ChatSettings(
        [
            hf_api,
            pc_api,
            model_select,
            query_type,
            pdf_select,
        ]
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    query = msg.content
    print(cl.user_session.get("paths"))
    if cl.user_session.get("task") == "internet_query" and torch.cuda.is_available():
        msg = cl.Message(content="")
        await msg.send()
        prompt = internet_query_obj.main(query=query)
        input_ids = tokenizer(prompt, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Generate an output of tokens
        outputs = llm_model.generate(
            **input_ids,
            temperature=0.7,
            do_sample=True,
            max_new_tokens=512,
        )
        output_text = tokenizer.decode(outputs[0])
        output_text = (
                output_text.replace(prompt, "")
                .replace(prompt, "")
                .replace("<|begin_of_text|>", "")
            )
        msg = cl.Message(content=output_text)
        await msg.send()
    if not cl.user_session.get("api_status"):
        msg1 = cl.ErrorMessage(content=cl.user_session.get("err"))
        await msg1.send()
        return
    if query == ".pdf":
        await load_pdf_to_pinecone()
        return
    if query == ".data":
        await load_pdf()
        return
    else:
        try:
            obj = cl.user_session.get("searcher")
            msg = cl.Message(content="")
            await msg.send()

            msg = cl.Message(
                content=obj.chainlit_prompt(
                    query,
                    cl.user_session.get("task"),
                    cl.user_session.get("pc"),
                    cl.user_session.get("hf"),
                    cl.user_session.get("slot"),
                    cl.user_session.get("model"),
                )
            )
            await msg.send()

        except Exception as e:
            msg = cl.Message(content="Error: " + str(e))


async def load_pdf_to_pinecone():
    index_name = cl.user_session.get("slot")
    if index_name in ["chat", "nutrition"]:
        msg = cl.Message(content=f"Can't Upload in {index_name}!")
        await msg.send()
        return
    res = await cl.AskActionMessage(
        content="Ready to Upload!",
        actions=[
            cl.Action(name="continue", value="continue", label="✅ Continue"),
            cl.Action(name="cancel", value="cancel", label="❌ Cancel"),
        ],
    ).send()

    if res and res.get("value") == "continue":
        files = await cl.AskFileMessage(
            content=f"Please upload a PDF file to load in {index_name}, `Please Wait!`",
            accept=["application/pdf"],
            max_size_mb=250,
            timeout=180,
        ).send()

        file = files[0]
        dict = cl.user_session.get("paths")
        dict[index_name] = file.path
        obj1 = cl.user_session.get("searcher")
        cl.user_session.set("paths", dict)
        obj = DataIngestionPipeLine()
        obj.load_to_pincone(cl.user_session.get("pc"), file.path, index_name)
        obj1.query_answer.setup_pd(dict)
        msg = cl.Message(content=f"Uploaded to {index_name}! `Start messaging...`")
        await msg.send()


async def load_pdf():
    index_name = cl.user_session.get("slot")
    if index_name in ["chat", "nutrition"]:
        msg = cl.Message(content=f"Can't Upload in {index_name}!")
        await msg.send()
        return
    res = await cl.AskActionMessage(
        content="Ready to Upload!",
        actions=[
            cl.Action(name="continue", value="continue", label="✅ Continue"),
            cl.Action(name="cancel", value="cancel", label="❌ Cancel"),
        ],
    ).send()

    if res and res.get("value") == "continue":
        files = await cl.AskFileMessage(
            content=f"Please upload a PDF file to load in {index_name}, `Please Wait!`",
            accept=["application/pdf"],
            max_size_mb=250,
            timeout=180,
        ).send()

        file = files[0]
        dict = cl.user_session.get("paths")
        dict[index_name] = file.path
        cl.user_session.set("paths", dict)
        obj = DataIngestionPipeLine()
        obj1 = cl.user_session.get("searcher")
        obj.store_tokens(cl.user_session.get("pc"), file.path, index_name)
        obj1.query_answer.setup_pd(dict)
        msg = cl.Message(content=f"Uploaded to {index_name}! `Start messaging...`")
        await msg.send()


@cl.on_settings_update
async def verify_keys(settings):
    if not cl.user_session.get("api_status"):
        pc = Pinecone(settings["pc_key"])
        try:
            print(pc.list_indexes().names())
            cl.user_session.set(
                "searcher", SearchAnswerPipeline(settings["pc_key"], llm_model)
            )
            cl.user_session.set("pc", settings["pc_key"])
        except:
            message = "Invalid Pinecone Key"
            cl.user_session.set("err", message)
            return
        import requests

        API_TOKEN = settings["hf_key"]
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response

        response = query("Can you please let us know more details about your ")
        if response.status_code == 400:
            message = "Invalid Hugging Face Token"
            cl.user_session.set("err", message)
            return
        cl.user_session.set("hf", settings["hf_key"])
        cl.user_session.set("api_status", True)
        msg = cl.Message(content="All Set 😎!")
        await msg.send()
    cl.user_session.set("model", settings["Model"])
    cl.user_session.set("slot", settings["Pdf"])
    cl.user_session.set("task", settings["Query"])
