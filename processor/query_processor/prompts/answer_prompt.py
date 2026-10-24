# processor/query_processor/prompts/answer_prompt.py

ANSWER_PROMPT = """你是一个智能助手，请根据参考内容回答用户的问题。

要求：
1. 尽量基于【参考内容】和【用户问题】作答，不要编造不存在的事实。
2. 只输出文字答案。禁止输出【图片】区块、图片URL或自行构造任何图片地址；图片由系统从本地知识切片中单独提取和展示。

【参考内容】
{context}

【历史对话】
{history}

【相关商品/实体】
{item_names}

【用户问题】
{question}

请回答："""