from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["疾病", "药品", "症状", "食谱", "人口群体", "器官/组织", "科室", "细菌/病毒/寄生虫"]

PROMPTS["entity_extraction"] = """
---目的---
给定一段文本和一份文本中可能包含的实体类型列表，从文本中识别出这些类型的所有实体以及识别实体之间的所有关系，不要有所遗漏。
使用{language}作为输出语言。

---步骤---
1. 识别所有实体。对于每个已识别的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。
- entity_type：实体类型： 以下类型之一： [{entity_types}]；
- entity_description： 实体描述： 实体在该段文本中的全面描述。
将每个实体格式化为以下形式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1确定的实体中，找出明确的所有(source_entity, target_entity) 对，对于每一对相关实体，提取以下信息：
- source_entity: ：源实体的名称，步骤1中确定的实体名称；
- target_entity：目标实体的名称，步骤1中确定的实体名称；
- relationship_description: 解释源实体和目标实体相互关联的原因；
- relationship_strength: 表示源实体和目标实体之间关系强度的数值分值，0-10之间；
- relationship_keywords: 一个或多个关键词，概括关系的性质，侧重于具体细节。
将每个关系格式化为以下形式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 找出概括全文主要概念、主题或话题的关键词。这些关键词应能捕捉到文件中的总体思想。
关键词的格式为：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}为输出语言，将步骤1和2中识别出的所有实体和关系以单个列表的形式输出。使用 **{record_delimiter}** 作为列表分隔符。

5. 当结束时，输出{completion_delimiter}

######################
---例子---
######################
{examples}

#############################
---真实数据---
######################
实体类型: [{entity_types}]
文本内容:
{input_text}
######################
输出:"""



PROMPTS["entity_extraction_examples"] = [
"""例子1:
实体类型: ["疾病", "药品", "症状", "食谱", "人口群体", "器官/组织", "科室", "细菌/病毒/寄生虫"]
文本内容:
```
肺-胸膜阿米巴病:肺-胸膜阿米巴病是溶组织阿米巴原虫感染所致的肺及胸膜化脓性炎症,肝原性病变多发生在右下肺,血源性则多为两肺多发病变.科室:呼吸内科.常用药品:替硝唑片,甲硝唑片
```
输出：  
("entity"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"疾病"{tuple_delimiter}"肺-胸膜阿米巴病是溶组织阿米巴原虫感染所致的肺及胸膜化脓性炎症"){record_delimiter}
("entity"{tuple_delimiter}"阿米巴原虫"{tuple_delimiter}"细菌/病毒/寄生虫"{tuple_delimiter}"感染肺-胸膜阿米巴病的寄生虫"){record_delimiter}
("entity"{tuple_delimiter}"肺"{tuple_delimiter}"器官/组织"{tuple_delimiter}"肺-胸膜阿米巴病的主要靶器官"){record_delimiter}
("entity"{tuple_delimiter}"胸膜"{tuple_delimiter}"器官/组织"{tuple_delimiter}"肺-胸膜阿米巴病的主要靶器官"){record_delimiter}
("entity"{tuple_delimiter}"肝原性病变"{tuple_delimiter}"疾病"{tuple_delimiter}"肺-胸膜阿米巴病的主要病变类型"){record_delimiter}
("entity"{tuple_delimiter}"血源性病变"{tuple_delimiter}"疾病"{tuple_delimiter}"肺-胸膜阿米巴病的主要病变类型"){record_delimiter}
("entity"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"科室"{tuple_delimiter}"肺-胸膜阿米巴病所属科室"){record_delimiter}
("entity"{tuple_delimiter}"替硝唑片"{tuple_delimiter}"药品"{tuple_delimiter}"肺-胸膜阿米巴病患者常用药品"){record_delimiter}
("entity"{tuple_delimiter}"甲硝唑片"{tuple_delimiter}"药品"{tuple_delimiter}"肺-胸膜阿米巴病患者常用药品"){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"阿米巴原虫"{tuple_delimiter}"溶组织阿米巴原虫感染可能会导致肺-胸膜阿米巴病"{tuple_delimiter}"感染,致病原因"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"肺"{tuple_delimiter}"溶组织阿米巴原虫感染所致的肺化脓性炎症可能是肺-胸膜阿米巴病"{tuple_delimiter}"靶向器官,作用组织"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"胸膜"{tuple_delimiter}"溶组织阿米巴原虫感染所致的胸膜化脓性炎症可能是肺-胸膜阿米巴病"{tuple_delimiter}"靶向器官,作用组织"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"肝原性病变"{tuple_delimiter}"肝原性病变是肺-胸膜阿米巴病的主要病变类型，主要作用器官是右下肺"{tuple_delimiter}"病变类型"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"血源性病变"{tuple_delimiter}"血源性病变是肺-胸膜阿米巴病的主要病变类型，主要作用器官是两肺"{tuple_delimiter}"病变类型"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"呼吸内科是肺-胸膜阿米巴病所属的科室，疑似患有该疾病的人群应于呼吸内科进行诊断治疗"{tuple_delimiter}"所属科室"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺"{tuple_delimiter}"肝原性病变"{tuple_delimiter}"肺-胸膜阿米巴病肝原性病变的主要作用器官是右下肺"{tuple_delimiter}"靶向器官"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"肺"{tuple_delimiter}"血源性病变"{tuple_delimiter}"肺-胸膜阿米巴病血源性病变的主要作用器官是两肺"{tuple_delimiter}"靶向器官"{tuple_delimiter}3){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"替硝唑片"{tuple_delimiter}"替硝唑片是肺-胸膜阿米巴病的常用药品，但是否需要需要参考医生建议"{tuple_delimiter}"常用药品"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺-胸膜阿米巴病"{tuple_delimiter}"甲硝唑片"{tuple_delimiter}"甲硝唑片是肺-胸膜阿米巴病的常用药品，但是否需要需要参考医生建议"{tuple_delimiter}"常用药品"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"肺-胸膜阿米巴病简介,肺-胸膜阿米巴病药物,肺-胸膜阿米巴病科室"){completion_delimiter}
###############""",
"""例子2:
实体类型: ["疾病", "药品", "症状", "食谱", "人口群体", "器官/组织", "科室", "细菌/病毒/寄生虫"]
```
肺泡蛋白沉着症:肺泡蛋白沉着症是一种原因未明的少见疾病.症状:肺泡炎症,胸痛,乏力.科室:呼吸内科
```
输出：
("entity"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"疾病"{tuple_delimiter}"肺泡蛋白沉着症是一种原因未明的少见疾病"){record_delimiter}
("entity"{tuple_delimiter}"肺泡"{tuple_delimiter}"器官/组织"{tuple_delimiter}"肺泡蛋白沉着症的主要靶器官"){record_delimiter}
("entity"{tuple_delimiter}"肺泡炎症"{tuple_delimiter}"症状"{tuple_delimiter}"肺泡蛋白沉着症的症状"){record_delimiter}
("entity"{tuple_delimiter}"胸痛"{tuple_delimiter}"症状"{tuple_delimiter}"肺泡蛋白沉着症的症状"){record_delimiter}
("entity"{tuple_delimiter}"乏力"{tuple_delimiter}"症状"{tuple_delimiter}"肺泡蛋白沉着症的症状"){record_delimiter}
("entity"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"科室"{tuple_delimiter}"肺泡蛋白沉着症所属科室"){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"肺泡"{tuple_delimiter}"肺泡是肺泡蛋白沉着症的主要靶向器官"{tuple_delimiter}"靶向器官"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"肺泡炎症"{tuple_delimiter}"肺泡炎症是一种疾病，同时也是肺泡蛋白沉着症的呈现症状，即出现肺泡炎症的人群有一定的可能性患有肺泡蛋白沉着症"{tuple_delimiter}"可能症状"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"胸痛"{tuple_delimiter}"胸痛是肺泡蛋白沉着症的呈现症状，即出现胸痛的人群有一定的可能性患有肺泡蛋白沉着症"{tuple_delimiter}"可能症状"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"乏力"{tuple_delimiter}"乏力是肺泡蛋白沉着症的呈现症状，即出现乏力的人群有一定的可能性患有肺泡蛋白沉着症"{tuple_delimiter}"可能症状"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"肺泡蛋白沉着症"{tuple_delimiter}"呼吸内科"{tuple_delimiter}"呼吸内科是肺泡蛋白沉着症所属的科室，疑似患有该疾病的人群应于呼吸内科进行诊断治疗"{tuple_delimiter}"所属科室"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"肺泡炎症"{tuple_delimiter}"肺泡"{tuple_delimiter}"肺泡是肺泡蛋白沉着症的主要靶向器官，同时也是肺泡炎症的主要靶向器官"{tuple_delimiter}"靶向器官"{tuple_delimiter}2){record_delimiter}
("content_keywords"{tuple_delimiter}"肺泡蛋白沉着症简介,肺泡蛋白沉着症症状,肺泡蛋白沉着症科室"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """
你作为智能助手，需要根据以下实体描述信息进行最终描述信息的整合，具体要求如下：
1、合并所有相关描述内容，但需要保留全部的描述关键信息
2、描述间可能存在矛盾点，需要通过你的专业知识进行辨识和解决
3、采用第三人称客观叙述，确保包含体名称作为主语
5、使用{language}语言输出

#######
---数据---
实体：{entity_name}
描述列表：{description_list}
#######
输出:
"""

PROMPTS["entity_continue_extraction"] = """
注意：前次提取存在明显遗漏实体和关系，请严格按以下流程进行补充。
---步骤---
1. 识别所有实体。对于每个已识别的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。
- entity_type：实体类型： 以下类型之一： [{entity_types}]；
- entity_description： 实体描述： 实体在该段文本中的全面描述。
将每个实体格式化为以下形式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤1确定的实体中，找出明确的所有(source_entity, target_entity) 对，对于每一对相关实体，提取以下信息：
- source_entity: ：源实体的名称，步骤1中确定的实体名称；
- target_entity：目标实体的名称，步骤1中确定的实体名称；
- relationship_description: 解释源实体和目标实体相互关联的原因；
- relationship_strength: 表示源实体和目标实体之间关系强度的数值分值，0-10之间；
- relationship_keywords: 一个或多个关键词，概括关系的性质，侧重于具体细节。
将每个关系格式化为以下形式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 找出概括全文主要概念、主题或话题的关键词。这些关键词应能捕捉到文件中的总体思想。
关键词的格式为：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以{language}为输出语言，将步骤1和2中识别出的所有实体和关系以单个列表的形式输出。使用 **{record_delimiter}** 作为列表分隔符。

5. 当结束时，输出{completion_delimiter}

---输出---
使用相同的格式将它们添加到下面：\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---目标---
似乎仍有部分实体被遗漏。

---输出---
仅回答 YES 或 NO，判断是否还有需要添加的实体。
""".strip()

PROMPTS["fail_response"] = ("抱歉，我无法回答该问题。没有相应的关键词，没有相关的上下文")

PROMPTS["keywords_extraction"] = """
---角色---
你是一名得力的关键词提取助手，负责从用户的查询和对话历史中识别关键词，包含实体关键词和关系关键词。

---目的---
根据查询和对话历史，列出实体关键词和症状关键词。
症状关键词来源于‘对话历史’，侧重于对话历史中用户提及的症状实体；
实体关键词来源于‘当前查询’，侧重于查询文本中与疾病相关的实体，是下列实体中的一部分：["药品","食谱","器官","组织","科室"]。


---说明---
-提取关键词时，会同时考虑当前查询和相关对话的历史记录
 -以 JSON 格式输出关键词，它将由 JSON 解析器解析，不要在输出中添加任何额外内容
 -JSON 应包含两个键：
  - 用于症状关键词的 "low_level_keywords"
  - 用于实体关键词的 "high_level_keywords"


######################
---例子---
######################
{examples}

#############################
---真实数据---
######################
对话历史：
{history}

当前查询： 
{query}
######################
输出应为人类文本，而非 unicode 字符。保持与 'query' 相同的语言。
输出:

"""

PROMPTS["keywords_extraction_examples"] = [
"""案例1
对话历史：
"用户：医生，我这两天突然胸口疼，呼吸也有点困难。
医生：您好，请问这种胸痛是持续性的还是间歇性的？疼痛时有没有伴随咳嗽或其他症状？
用户：是一阵一阵的疼，深呼吸时特别明显，还有点干咳。
医生：是否有外伤史或近期剧烈运动（如提重物、打球等）？
用户：昨天打篮球时确实被撞了一下胸口，当时没在意，但晚上开始不舒服。"

当前查询： 
"你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病、诊断原因、疾病概述、就医科室以及推荐食谱等生活习惯。"

################
输出:
{
  "high_level_keywords": ["食谱","器官","组织","科室"]，
  "low_level_keywords": ["间歇性胸痛","呼吸困难","干咳","深呼吸加剧性胸痛","胸口撞击伤"]

}
##########################""",

"""案例2:
对话历史：
"用户：医生，我父亲最近记忆力越来越差，经常忘记刚发生的事情，脾气也变得很奇怪，我们有点担心。
医生：您好，请问您父亲今年多大年纪？这种情况大概持续多久了？
用户：他72岁了，最近半年特别明显，有时候连家里的路都认不清，还总说看到不存在的东西。
医生：除了记忆力减退和幻觉，他是否还有以下情况？比如情绪不稳定、语言表达困难、或者日常生活能力下降（如穿衣、吃饭困难）？
用户：对！他有时候会突然发脾气，说话也经常词不达意，最近连系扣子都变得很困难。"

当前查询： 
"你是一名三甲医院的医生，请根据你和用户之间的对话并结合提供数据库，向用户提供诊断意见，诊断意见包括：可能性的疾病、诊断原因、疾病概述、推荐药物以及推荐食谱等生活习惯。"

################
输出:
{
  "high_level_keywords": ["食谱","器官","组织","药品"],
  "low_level_keywords": ["记忆力差","脾气奇怪","幻觉","持续半年","记忆力减退","情绪不稳定","语言表达困难","日常生活能力下降"]
}
##########################""",
]

PROMPTS[
    "similarity_check"
] = """
请分析这两个问题的相似之处:

问题1: {original_prompt}
问题2: {cached_prompt}

请评估这两个问题在语义上是否相似，问题2的答案是否可以用来回答问题 1，并直接给出0到1之间的相似度分数。

相似度评分标准：
0：完全不相关或答案不能重复使用，包括但不限于：
   - 问题的主题不同
   - 问题中提到的症状不同
   - 问题中提到的症状持续时间不同
   - 问题中提到的具体生活习惯不同
   - 问题中提到的需要回答的关键词不同
   - 问题中提到的对话信息不同
1：所提及的所有内容均相同，答案可直接重复使用
0.5：仅部分内容相关，答案需要修改才能使用
只返回 0-1 之间的数字，不含任何附加内容。
"""


PROMPTS["knowledge"] = """
以下为你的基础知识，之后的回答都需要严格依据知识库进行，不可以脱离知识库进行回答，不可以回答知识库内不存在的内容。

---知识库---
{knowledge}
"""


PROMPTS["list_all_diseases"] = """
你是一位专业的医疗辅助诊断系统，现在需要基于你的知识库即你的基础知识，列出与用户症状最相关的前5个疾病名称，即最多列出5个知识库内的疾病。请严格按照以下要求执行：

---任务要求---

首先全面分析用户描述的症状特征，包括症状性质、部位、持续时间等关键信息

从知识库中筛选出所有可能与这些症状相关的疾病

根据以下标准对疾病进行相关性排序：
a) 症状匹配度 - 疾病核心症状与用户描述症状的吻合程度
b) 特异性 - 该症状对某种疾病的特异性强弱
c) 常见度 - 该疾病在人群中的普遍性
输出排序后的疾病列表，相关性最强的排在最前面
如果相关疾病超过5个，则只需要列出最相关的5个
严格保证疾病名称与知识库记录完全一致

---输出格式---
[
"疾病名称1",
"疾病名称2",
"疾病名称3",
...
]

---注意事项---
只输出疾病名称，不要包含任何解释、症状描述或其他信息
确保所有疾病名称都来自知识库，不要虚构
严格按照Python列表格式输出
不要添加任何非列表内容，包括引导性文字

---用户症状---
{keywords}
"""


PROMPTS["extract_knowledge_ill"] = """
作为医疗知识整合系统，请根据提供的疾病列表从知识库中提取对应生活习惯建议，按以下要求处理：

---输入处理---
若疾病名称存在俗称与知识库内表述不一致（如"中风"），自动转换为知识库内表述即标准术语（如"脑卒中"）

---知识提取规则---
所提取的知识尽可能包含疾病的以下相关内容：{data}
必须提取疾病列表中全部疾病的相关内容，不可以有任何的遗漏

---输出模板---
1.标准疾病名称A：
    1.1饮食建议：……
    1.2用药建议：……
    1.3就诊指引：……
2.标准疾病名称A：
    2.1用药建议：……
    2.2就诊指引：……
……

---疾病列表---
{keywords}

"""


PROMPTS["low_level_keywords_system"] = """
---角色---
您是一位专业的三甲医院医生，以下为你的基础知识，之后的回答都需要着重依据知识库进行，但可以涉及到知识库内没有的疾病。你需要根据用户提供的对话历史,症状关键词，检索知识库内信息，进行用户问题回答即推理出用户可能患有的任何疾病的中文名称，输出内容需要严格符合输出格式即python编程语言的list列表格式。

---目的---
着重考虑症状关键词和对话历史，根据知识库严格生成符合输出格式的回应。

---输出格式---
输出格式为python编程语言的list格式，即中文疾病实体外侧需要使用英文单引号''包裹，中文疾病实体之间需要使用英文逗号,分隔，所有的中文疾病实体都需要存储在英文中括号[]之内。除此之外不需要有其他任何多余内容。

---知识库---
{context_data}

"""


PROMPTS["low_level_keywords_prompt"] = """
你好，请根据以下信息，基于知识库内的基础知识，使用中文，列出可能患有的疾病的中文名称。输出格式为python编程语言的list格式[]。\n

---对话历史---
{history}

---症状关键词---
{keywords}

"""

PROMPTS["high_level_keywords_system"] = """
---角色---
您是一位专业的三甲医院医生，以下为你的基础知识，之后的回答都需要着重依据知识库进行，但可以涉及到知识库内没有的疾病。你需要基于知识库内的基础知识，根据用户提供的：对话历史，症状关键词，可能性的疾病，在知识库内进行相关检索，按照回答规则进行用户问题回答。

---目的---
考虑用户提供的对话历史，当前查询，症状关键词，可能性的疾病，根据已知知识库和回答规则生成简明的响应。总结所提供知识库中的所有信息，并纳入与知识库相关的常识，不包含知识库未提供的信息。

---回答规则---
- 目标格式和长度：{response_type}；
- 使用markdown格式和适当的章节标题；
- 使用用户提问的相同语言进行回应；
- 确保答复与对话历史保持连续性；
- 在结尾处的 "References" 部分列出最多 5 个，最少1个最重要的参考来源，并明确呈现这些来源所提供的具体内容，便于用户参考；
- 如果不知道答案，请直说；
- 不要胡编乱造。请勿包含知识库未提供的信息；
- 请将知识库内与回答相关的内容追加在你回答内容的最后，方便用户进行验证。

---知识库---
{content_data}"""

PROMPTS["high_level_keywords_prompt"] = """
你是一名三甲医院的医生，请基于知识库内的基础知识，根据你和用户之间的对话，向用户提供诊断意见，诊断意见包括：可能性的疾病、疾病概述以及宜吃和忌吃食物等生活习惯。最后请将知识库内与回答相关的内容追加在回答内容的最后，方便进行验证。

---对话历史---
{history}

---症状关键词---
{keywords}

---可能性的疾病---
{ills}
"""



# PROMPTS["rag_response"] = """
# ---角色---
# 您是一位专业的三甲医院医生，需要根据下方提供的对话历史和知识库，按照回答规则进行用户问题回答。

# ---目的---
# 根据知识库生成简洁回应，并遵循回应规则，同时考虑对话历史和当前查询。参考知识库中所有信息，并整合与查询以及对话历史相关的信息，不要包含知识库未提供的信息。

# 处理带时间戳的关系时：
# 1.每个关系都有一个"created_at"时间戳，表示我们获取该知识的时间；
# 2.遇到冲突关系时，需同时考虑语义内容和时间戳；
# 3.不要自动优先选择最新创建的关系 - 需根据上下文进行判断；
# 4.对于时间敏感的查询，优先考虑内容中的时间信息，再考虑创建时间戳。

# ---对话历史---
# {history}

# ---知识库---
# {context_data}

# ---回答规则---
# - 目标格式和长度：{response_type}；
# - 使用markdown格式和适当的章节标题；
# - 使用用户提问的相同语言进行回应；
#  -确保答复与对话历史保持连续性；
#  -在结尾处的 "References" 部分列出最多 5 个最重要的参考来源。
#  -明确指出每个来源是来自知识图谱 (KG) 还是矢量数据 (DC)，如有文件路径，请按以下格式提供： [KG/DC] 文件路径；
#  -如果不知道答案，请直说；
#  -不要胡编乱造。请勿包含知识库未提供的信息。"""


# PROMPTS["naive_rag_response"] = """
# ---角色---

# 您是一位乐于助人的助手，正在回答用户提出的有关文档块的问题。

# ---目的---
# 根据文档块和回答规则生成简明的响应，同时考虑对话历史和当前查询。总结所提供文档块中的所有信息，并纳入与文档块相关的常识。不包含文档块未提供的信息。

# 处理带时间戳的关系时：
# 1.每个关系都有一个"created_at"时间戳，表示我们获取该知识的时间；
# 2.遇到冲突关系时，需同时考虑语义内容和时间戳；
# 3.不要自动优先选择最新创建的关系 - 需根据上下文进行判断；
# 4.对于时间敏感的查询，优先考虑内容中的时间信息，再考虑创建时间戳。

# ---对话历史---
# {history}

# ---文档块---
# {content_data}

# ---回答规则---

# - 目标格式和长度：{response_type}；
# - 使用markdown格式和适当的章节标题；
# - 使用用户提问的相同语言进行回应；
# - 确保答复与对话历史保持连续性；
# 在结尾处的 "References" 部分列出最多 5 个，最少1个最重要的参考来源，并明确呈现这些来源所提供的具体内容，便于用户参考；
# - 如果不知道答案，请直说；
# - 不要胡编乱造。请勿包含知识库未提供的信息。"""




# PROMPTS["mix_rag_response"] = """
# ---角色---

# 您是一位乐于助人的助手，正在回答用户有关数据源的询问。

# ---目的---

# 根据数据源和响应规则生成简明的响应，同时考虑对话历史和当前查询。数据源包括两部分： 知识图谱（KG）和文档块（DC）。总结提供的数据源中的所有信息，并纳入与数据源相关的常识。不要包含数据源未提供的信息。

# 处理带时间戳的关系时：
# 1.每个关系都有一个"created_at"时间戳，表示我们获取该知识的时间；
# 2.遇到冲突关系时，需同时考虑语义内容和时间戳；
# 3.不要自动优先选择最新创建的关系 - 需根据上下文进行判断；
# 4.对于时间敏感的查询，优先考虑内容中的时间信息，再考虑创建时间戳。

# ---对话历史---
# {history}

# ---数据来源---

# 1. 来自于知识图谱(KG):
# {kg_context}

# 2. 来自于文档块(DC):
# {vector_context}

# ---回答规则---

# - 目标格式和长度：{response_type}；
# - 使用markdown格式和适当的章节标题；
# - 使用用户提问的相同语言进行回应；
# - 确保答复与对话历史保持连续性；
# - 在结尾处的 "References" 部分列出最多 5 个，最少1个最重要的参考来源。明确指出每个来源是来自知识图谱 (KG) 还是矢量数据 (DC)，如有文件路径，请按以下格式提供： [KG/DC] 文件路径，并明确呈现这些来源所提供的具体内容，便于用户参考；
# - 按章节组织答案，侧重于答案的一个要点或方面；
# - 使用能反映内容的清晰、描述性的章节标题；
# - 如果不知道答案，请直说；
# - 不要胡编乱造。请勿包含知识库未提供的信息。"""