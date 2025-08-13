import os
import io
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI, Depends, Response
from fastapi.responses import StreamingResponse
from resbianque_models import ChatVO
import httpx
import json


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 默认使用0号显卡，避免Windows用户忘记修改该处
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定模型名称或路径
bianque_v2_model_name_or_path = "../../1_testmodel/ori/BianQue-2"
def load_bianque_v2_model():
    bianque_v2_model = AutoModel.from_pretrained(bianque_v2_model_name_or_path, trust_remote_code=True).half()
    # bianque_v2_model = T5ForConditionalGeneration.from_pretrained(bianque_v2_model_name_or_path)
    bianque_v2_model.to(device)
    print('bianque_v2 model Load done!')
    return bianque_v2_model
def load_bianque_v2_tokenizer():
    bianque_v2_tokenizer = AutoTokenizer.from_pretrained(bianque_v2_model_name_or_path, trust_remote_code=True)
    print('bianque_v2 tokenizer Load done!')
    return bianque_v2_tokenizer
bianque_v2_model = load_bianque_v2_model()
bianque_v2_tokenizer = load_bianque_v2_tokenizer()



async def answer(input_content, sample=True, bianque_v2_top_p=0.7, bianque_v2_temperature=0.95,
                 bianque_v1_top_p=1, bianque_v1_temperature=0.7):
    if not sample:
        response, history = bianque_v2_model.chat(bianque_v2_tokenizer, query=input_content, history=None,
                                                  max_length=4096, num_beams=1, do_sample=False,
                                                  top_p=bianque_v2_top_p, temperature=bianque_v2_temperature,
                                                  logits_processor=None)
    else:
        response, history = bianque_v2_model.chat(bianque_v2_tokenizer, query=input_content, history=None,
                                                  max_length=4096, num_beams=1, do_sample=True,
                                                  top_p=bianque_v2_top_p, temperature=bianque_v2_temperature,
                                                  logits_processor=None)
    return response


app = FastAPI()
client = httpx.AsyncClient()

async def res_generator(output):
    is_end = "false"
    index = 0
    for a in output:
        if index == len(output)-1:
            is_end = "true"
        res = f'data: {{"result":"{a}","is_end":{is_end}}}\n'.encode('utf-8')
        index = index + 1
        yield res
      

@app.post("/bianque2")
async def chat(chat_vo: ChatVO):
    content = chat_vo.content
    output = await answer(input_content = content)
    output = output.replace("\n","")
    return StreamingResponse(res_generator(output), media_type="text/event-stream")
