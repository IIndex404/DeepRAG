from  dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()
@dataclass
class RerankerConfig:
    text_rerank_api_key:str #调用百炼Rerank模型需要的API Key
    text_rerank_model:str  #使用哪个Rerank模型
    text_rerank_instruct:str #告诉Rerank模型按照什么要求判断文档和问题是否相关
reranker_config = RerankerConfig(
    text_rerank_api_key = os.getenv("OPENAI_API_KEY"),
    text_rerank_model = os.getenv("TEXT_RERANK_MODEL"),
    text_rerank_instruct = os.getenv("TEXT_RERANK_INSTRUCT"),
)