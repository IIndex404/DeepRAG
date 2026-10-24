ITEM_NAME_SYSTEM_PROMPT ="""
你是一个专业的商品名称识别模型，请根据提供的信息识别商品名称。
"""
# 用户提示词模板：后面会把 file_title 和 context 填进去
ITEM_NAME_USER_PROMPT_TEMPLATE ="""
请从以下信息中识别出商品名称与型号：

文件名：
{file_title}

正文切片：
{context}

要求：
1. 尽量返回完整的商品名称，包括品牌、型号和商品名称；
2. 只返回商品名称，不要添加解释；
3. 如果无法识别，请返回空字符串。
"""