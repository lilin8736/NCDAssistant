import os
import io
import torch
from transformers import AutoModel, AutoTokenizer,T5Tokenizer, T5ForConditionalGeneration,AutoModelForCausalLM
from fastapi import FastAPI, Depends, Response
from fastapi.responses import StreamingResponse
from res_models import ChatVO
import httpx
import json
from tqdm import trange
import ast
from gensim.models import KeyedVectors,word2vec
import itertools
import jieba
from sklearn.metrics.pairwise import cosine_similarity
import xiangsi as xs
import requests
import pandas as pd
import logging
import asyncio
import nest_asyncio
nest_asyncio.apply()
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from typing import Any, AsyncIterator
from lightrag.operate import chunking_by_token_size


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 默认使用0号显卡，避免Windows用户忘记修改该处
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
knowledge=pd.read_csv('../data/total.csv') #疾病表格

with open('med_sys.txt','r') as f:
    data_line = f.readlines()
data_line = [i[:-1] for i in data_line]
with open('med.txt','r') as f:
    data_med = f.readlines()
data_med = [i[:-1] for i in data_med] #数据集加载
WORKING_DIR = "./lightragpag/selfLightRAG/workapcesworkapces"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


"""模型加载与访问"""
#加载模型-小金刚
chimpanzeestf_path = "../../1_testmodel/stf/output1010/MiniCPM3_4B_5epoch"
bianque_v2_model_name_or_path = "../../1_testmodel/ori/BianQue-2"
chimpanzee_path = "../../1_testmodel/ori/MiniCPM3-4B"
chimpanzeestf_tokenizer = AutoTokenizer.from_pretrained(chimpanzeestf_path,trust_remote_code=True)
chimpanzeestf_model = AutoModelForCausalLM.from_pretrained(chimpanzeestf_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
chimpanzee_tokenizer = AutoTokenizer.from_pretrained(chimpanzee_path,trust_remote_code=True)
chimpanzee_model = AutoModelForCausalLM.from_pretrained(chimpanzee_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
bert_tokenizer = BertTokenizer.from_pretrained('./bert_base_chinese')
bert_model_name = 'bert_base_chinese'

async def initialize_rag():
    rag = LightRAG(
            working_dir=WORKING_DIR,
            chunking_func=chunking_by_token_size,
            llm_model_func=ollama_model_complete,
            llm_model_name="deepseek-r1:32b8192",
            llm_model_max_async=4,
            llm_model_max_token_size=131072,
            llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 8192}},
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts, embed_model="bge-m3:latest8192", host="http://localhost:11434"
                ),
            ),
        )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag
rag = asyncio.run(initialize_rag()) #知识图谱检索内容

#bert科室分类
class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return pooled_output
bert_model = BERTClassifier() #加载模型
bert_model.to(device)

class modelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(modelClassifier, self).__init__()
        self.num_labels = num_labels
        self.embed = 768
        self.hidden_size = 256
        self.num_layers = 3 # lstm层数
        self.dropout = 1.0  # 随机失活
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.lstm = nn.LSTM(self.embed , self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embed, self.num_labels)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = self.fc(out)
        out = out.squeeze(dim=1)
        return out
    
Classifiermodel_path = './model.pth'
Classifiermodel_params = torch.load(Classifiermodel_path)
Classifiermodel = modelClassifier(17)
Classifiermodel.load_state_dict(Classifiermodel_params)
Classifiermodel.to(device)
sys_classes = ['急诊科', '外科', '皮肤性病科', '营养科', '精神科', '内科', '男科', '心理科', '儿科', '生殖健康', '妇产科', '肿瘤科', '其他科室', '肝病', '中医科', '五官科', '传染科']

#根据症状分科室
async def predict(text):
    text = '最近我感觉有点：'+text+'。应该去挂什么科？'
    Classifiermodel.eval()
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = bert_model(input_ids, attention_mask).to(device)
    with torch.no_grad():
        outputs = outputs[0]
        outputs = outputs.unsqueeze(0)
        outputs = outputs.unsqueeze(0)
        outputs = Classifiermodel(outputs).to(device)
        softmax = torch.nn.Softmax(dim=1)
        predictions = softmax(outputs)
        predictions = predictions.cpu().tolist()[0]
        sorted_pairs = sorted(zip(sys_classes, predictions), key=lambda pair: pair[1], reverse=True)
        sorted_classes, predictions = zip(*sorted_pairs)
        sorted_classes = list(sorted_classes)
        predictions= list(predictions)
    result = f'根据症状,患者的疾病可能属于{sorted_classes[0]}、{sorted_classes[1]}或{sorted_classes[2]} 科室,概率分别为:{predictions[0]}、{predictions[1]}和{predictions[2]}。'
    return result

#访问扁鹊2模型
async def bianque2(content):
    url = "http://127.0.0.1:9295/bianque2"  
    data = {"content": f"{content}"}  
    response = requests.post(url, json=data)
    response = response.text
    result_str = ""
    for line in response.splitlines():  
        if line.startswith('data:'):   
            json_str = line.split('data: ')[1]
            if "is_end" not in json_str:
                continue
            data = json.loads(json_str)  
            result_str += data['result']
    return result_str

#访问小金刚模型
async def chimpanzee(prompt,content):
    device = "cuda"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]
    inputs = chimpanzee_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    model_outputs = chimpanzee_model.generate(
        inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.7,
        pad_token_id=chimpanzee_tokenizer.eos_token_id  #设置 `pad_token_id` 为 `eos_token_id`
    ).to(device)
    # 解码生成的文本
    output_token_ids = [
        model_outputs[i][len(inputs[i]):] for i in range(len(inputs))
    ]
    responses = chimpanzee_tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    responses = responses.replace("\n\n","")
    return responses

#访问微调小金刚模型
async def chimpanzeestf(prompt,content):
    device = "cuda"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]
    inputs = chimpanzeestf_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    model_outputs = chimpanzeestf_model.generate(
        inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.7,
        pad_token_id=chimpanzee_tokenizer.eos_token_id  #设置 `pad_token_id` 为 `eos_token_id`
    ).to(device)
    # 解码生成的文本
    output_token_ids = [
        model_outputs[i][len(inputs[i]):] for i in range(len(inputs))
    ]
    responses = chimpanzeestf_tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    responses = responses.replace("\n\n","")
    return responses






"""访问流程"""
#根据最终疾病获得补充信息
async def get_kg_all(diseasessymptoms):
    triple =[]
    for i in range(len(knowledge)):
        name =knowledge['疾病名称'][i]
        synopsis =knowledge['简介'][i]
        symptoms = knowledge['症状'][i]
        combine =knowledge['并发症'][i]
        jzks =knowledge['就诊科室'][i]
        zlfs =knowledge['治疗方式'][i]
        zlzq =knowledge['治疗周期'][i]
        zyl =knowledge['治愈率'][i]
        cyyp =knowledge['常用药品'][i]
        zlfy =knowledge['治疗费用'][i]
        cause =knowledge['病因'][i]
        yffs =knowledge['预防方式'][i]
        zlgs =knowledge['治疗概述'][i]
        ycsw =knowledge['宜吃食物'][i]
        jcsw =knowledge['忌吃食物'][i]
        tjsp =knowledge['推荐食谱'][i]
        tjyp =knowledge['推荐药品'][i]
        ybjb =knowledge['医保疾病'][i]
        hbbl =knowledge['患病比例'][i]
        ygrq =knowledge['易感人群'][i]
        crfs =knowledge['传染方式'][i]
        triple.append((name,synopsis,symptoms,combine,jzks,zlfs,zlzq,zyl,cyyp,zlfy,cause,yffs,zlgs,ycsw, jcsw,tjsp,tjyp,ybjb,hbbl,ygrq,crfs))
    final = ""
    for name,synopsis,symptoms,combine,jzks,zlfs,zlzq,zyl,cyyp,zlfy,cause,yffs,zlgs,ycsw,jcsw,tjsp,tjyp,ybjb,hbbl,ygrq,crfs in triple:
        if isinstance(name,str) and name[:-4] in diseasessymptoms:
            synopsis = synopsis.replace('\n','')
            cause = cause.replace('\n','')
            final = final + f'疾病名称:{name};简介:{synopsis};症状:{symptoms};并发症:{combine};就诊科室:{jzks};治疗方式:{zlfs};治疗周期:{zlzq};治愈率:{zyl};常用药品:{cyyp};治疗费用:{zlfy};病因:{cause};预防方式:{yffs};治疗概述:{zlgs};宜吃食物:{ycsw};忌吃食物:{jcsw};推荐食谱:{tjsp};推荐药品:{tjyp};医保疾病:{ybjb};患病比例:{hbbl};易感人群:{ygrq};传染方式:{crfs};'
    return final


#可能症状可能性疾病-以及没有疾病的状况
async def kg_sym(history,base_information):
    example = '{"症状":["恶心","头晕"],"疾病":["内耳问题","颈椎病","贫血"]}'
    entity_prompt = "请从患者的基本信息以及患者和医生的对话中提取关键信息。请识别患者描述的所有症状，以及经过医生诊断可能患有的疾病。将症状和疾病分别列出，确保你的答案是准确无误的,且一定要按照输出示例以 JSON 格式进行输出，不需要添加格外内容。"
    entity_input = f"""下列为患者的基本信息:{base_information}\n
                    下列为患者和医生的对话历史:{history}\n
                    输出示例：{example}"""
    entity_responses = await chimpanzee(entity_prompt,entity_input) #小金刚提取症状和可能性疾病
    entity_responses = entity_responses.replace('“', '"').replace('”', '"').replace('，', ',').replace(']]', ']').replace('[[', '[').replace("'", '"').replace("‘", '"').replace("’", '"').replace("json", '').replace("```", '')
    print(entity_responses)
    entity_responses = json.loads(entity_responses)
    symptoms = entity_responses["症状"]
    candidate_diseases = entity_responses["疾病"]
    return symptoms,candidate_diseases

    
#评估是否满足
async def evalu(base_information, symptoms, diseases, extract_context):
    symptoms_str = '，'.join(symptoms)
    spartments = predict(symptoms_str) #部门
    example_des = '{“analysis”:..., “distribution”: {“动物皮肤病”: 0.27, “红斑”: 0.3, “皮炎”: 0.85}}'
    des_prompt = "你是一名专业医生，任务是根据患者的基本信息以及提供的症状信息对病人进行诊断。您将得到：可能的疾病科室、一份候选疾病清单和一些疾病的基础知识，您的任务是为患者提供详细的诊断分析和候选疾病的可信度分布。您需要首先分析患者的病情，并思考患者可能患有哪些候选疾病。如输出示例，以 JSON 格式输出分析和候选疾病的诊断置信度分布，请一定要按照输出示例格式进行输出。需要输出每一种候选疾病的诊断置信度分布，每种疾病的诊断置信度均在0-1之间，不需要总计为1，例如A疾病为0.8，B疾病可以为0.5。绝不可能患有的疾病置信度可以为0。"
    des_input = f"""输出示例：{example_des}\n
                    患者的基本信息:{base_information}\n
                    患者症状： {symptoms}\n
                    可能的疾病科室:{spartments}\n
                    候选疾病： {diseases}\n
                    疾病基础信息：{extract_context}"""
    
    des_output = await chimpanzee(des_prompt,des_input)#小金刚进行分析和置信度获取
    try:
        des_responses = json.loads(des_output)
        analysis = des_responses["analysis"] #分析
        distribution = des_responses["distribution"] #打分
    except json.JSONDecodeError:
    # 如果 des_output 不是有效的 JSON 字符串，则处理异常
    # 这里可以返回一个空字典或者打印错误信息
        print(des_output)
        des_responses = {}
        analysis = "" #分析
        distribution = {} #打分
    sorted_items = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    sim_de = dict(sorted_items)
    if list(sim_de.values())[0]>=0.5:
        ifxunhuan = False
    else:
        ifxunhuan = True
    return ifxunhuan,distribution


#获得最后的结果
async def get_end(distribution, query, hl_keywords ,ll_keywords):
    sorted_items = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    sim_de = dict(sorted_items)
    dise = list(sim_de.keys()) #疾病列表
    if len(dise)>5:
        dise = dise[:5] #获得前5个可能的疾病
        sim_de = dict(sorted(sim_de.items(), key=lambda item: item[0])[:5])
    
    history = "\n".join(hist.split("\n")[2:])
    conversation_history=[{"role": "user", "content": history}]
    results = rag.query(1, query, hl_keywords, dise, param=QueryParam(mode="ll_medical",conversation_history=conversation_history), ll_keywords=ll_keywords)    
    return results   



#获得疾病和症状列表
async def get_ills_sym(hist,result_hist,base_information):
    symptoms,candidate_diseases = await kg_sym(result_hist,base_information) #症状1和疾病1列表
    history = "\n".join(hist.split("\n")[2:])
    conversation_history=[{"role": "user", "content": history}]
    response = rag.query(0, hist, data_line, data_med, param=QueryParam(mode="ll_medical",conversation_history=conversation_history))
    ll_keywords = response[0].split(", ") #症状2列表
    hl_keywords = response[1].split(", ")
    ills = response[2].split(", ") #疾病2列表
    symptoms = ills + candidate_diseases #症状列表
    diseases = ll_keywords + symptoms #疾病列表
    return symptoms,diseases,hl_keywords


#获得疾病对应症状的补充信息
async def get_illsym(diseases):
    results = []
    data_line_n = [element.split(" ")[0] for element in data_line]
    for oo in diseases:
        result = [data_line[i] for i,element in enumerate(data_line_n) if oo==element]
        results = results + result
    else:
        results.append(oo + "症状:暂时没有症状信息。")
    result = []
    for oo in results:
        if oo not in result:
            result.append(oo)
    
    extract_context = "\n".join(result)
    return extract_context
    

async def answer(user_history, bot_history, base_information, sample=True, bianque_v2_top_p=0.7, bianque_v2_temperature=0.95,
                 bianque_v1_top_p=1, bianque_v1_temperature=0.7):
    if len(bot_history) > 0:
        context = "\n".join([f"病人：{user_history[i]}\n医生：{bot_history[i]}" for i in range(len(bot_history))])
        input_text = context + "\n病人：" + user_history[-1] + "\n医生："
    else:
        input_text = "病人：" + user_history[-1] + "\n医生："

    if len(bot_history) >= 4:
        #如果患者没有发出自己的疑问，则进行总结答案
        if '?' not in user_history[-1] and '？' not in user_history[-1]: # 最多允许问3个问题
            hist_ask = input_text
            hist = "\n你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病、疾病概述以及宜吃和忌吃食物等生活习惯。\n医患对话对话如下：\n" + input_text[:-4] #医患对话历史（不包含医生的结论）
            input_text =f"请依据患者的基本情况以及对话信息尽量给出一些建议的检查或者健康的生活及饮食习惯。\n患者的基本情况：{base_information}\n 对话信息:{input_text}"
            response = await bianque2(content=input_text) #bianque2 得出结论
            response = response.replace('医生：','').replace('患者：','')
            result_hist = hist + response #携带bianque2结论的历史记录
             #症状和候选疾病1 list格式
            symptoms,diseases,hl_keywords = get_ills_sym(hist,result_hist,base_information) #症状列表,疾病列表,查询关键词
            extract_context = get_illsym(diseases) #根据疾病补充症状信息
            ifxunhuan,distribution = await evalu(base_information, symptoms, diseases, extract_context) #判断置信度是否大于0.5

            # 顶多再问三次
            if ifxunhuan and len(bot_history)<=5:
                re_or_in = f"""假设你是一名经验丰富的智能家庭助手，你的目标是根据患者和医生的对话信息，通过一系列有目的的提问来收集必要的信息，包括但不限于了解患者的症状、病史、生活方式和其他相关因素。不需要给出任何的意见，只需要进行提问，且每次仅可以提出1-2个问题，不可以提问重复的问题。目前已知患者的基本情况为：{base_information}。下面为患者和医生的对话信息，请继续发起提问。\n\n对话信息:"""
                response = await chimpanzeestf(re_or_in,hist_ask) #继续提问
                return response
            
            response = await get_end(distribution, hist, hl_keywords, symptoms)
            response = response.replace('医生：','').replace('患者：','')
            
        else:
            symptoms,candidate_diseases = await kg_sym(input_text,base_information) #提取疾病
            final = await get_kg_all(candidate_diseases) #知识图谱补充信息
            response = await get_end_model(input_text,sim_de,final,False,base_information) #整合给出回答
    else:
        #至少提问4轮
        input_prompt =f"假设你是一名经验丰富的智能家庭助手，你的目标是根据患者和医生的对话信息，通过一系列有目的的提问来收集必要的信息，包括但不限于了解患者的症状、病史、生活方式和其他相关因素。不需要给出任何的意见，只需要进行提问，且每次仅可以提出1-2个问题，不可以提问重复的问题。目前已知患者的基本情况为：{base_information}。下面为患者和医生的对话信息，请继续发起提问。\n\n"
        response = await chimpanzeestf(input_prompt,input_text) #提问
        response = response.replace('医生：','').replace('患者：','')
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


@app.post("/chat")
async def chat(chat_vo: ChatVO):
    content = chat_vo.content
    user_id = chat_vo.user_id
    hist_labels = chat_vo.with_history
    base_information = chat_vo.base_information
    if "结束" in content:
        return StreamingResponse(res_generator("感谢您的咨询，希望小康的建议对您有所帮助。祝您健康幸福！"), media_type="text/event-stream")
    if hist_labels==0:
        user_history = []
        bot_history = []
    else:
        user_history = []
        bot_history = []
        url = "http://www.pahealthsys.cn:9999/admin/ai/history"
        data = {
            "user_id": user_id,
            "pwd": "xzhmueai",
            "size": 100
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = await client.post(url, headers=headers, json=data)
        j = response.json()["data"]
        for i in j:
            user_history.append(i["userContent"])
            bot_history.append(i["assistantContent"])
        user_history = list(reversed(user_history))
        bot_history = list(reversed(bot_history))
    # 构建user_history, bot_history
    user_history.append(content)
    print(len(bot_history))
    print(f"bot_history：{bot_history}")
    print(f"user_history：{user_history}")
    output = await answer(user_history, bot_history,base_information)
    output = output.replace("\n","")
    print(f"output:{output}")
    return StreamingResponse(res_generator(output), media_type="text/event-stream")