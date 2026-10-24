import dashscope
from dotenv import load_dotenv
from config.reranker_config import reranker_config
load_dotenv()
def rerank_documents(
        query:str,
        documents:list[str]
)->list[float]:
    """
    :param query: 用户的问题
    :param documents:  前面搜出来的一堆候选文档正文
    :return: 每篇文档对应的相关性分数例如：[0.95, 0.72, 0.21, 0.08]
    documents =
    ["HAK180烫金机的转印温度调节方法：接通电源后，按住设置键3秒，温度调到110度。",
    "今天中午去吃猪脚饭吧，这家店的味道很不错。",
    "烫金机的日常保养：定期清理加热板，避免灰尘堆积影响温度。",]
    """
    dashscope.api_key = (reranker_config.text_rerank_api_key)
    response = dashscope.TextReRank.call(
        model = reranker_config.text_rerank_model, #真正调用百炼的Rerank模型
        query = query,  #用户的问题
        documents = documents, #需要重新打分的所有候选文档
        top_n=len(documents),  #传进来多少篇文档,就让模型把多少篇都进行打分
        return_documents=False,  #只要分数就行
        instruct=reranker_config.text_rerank_instruct, #应该按照什么要求判断问题和文档是否相关
    )
    status_code = response.get("status_code")
    if status_code != 200: #拿这次API请求返回的状态码
        message = response.get("message") #拿百炼返回的具体报错信息
        raise RuntimeError( f"DashScope rerank 调用失败: {message}")
    results = response.output.get("results",[]) #拿模型真正返回的打分结果
    """
    results:
    {"index": 0, "relevance_score": 0.95},
    {"index": 1, "relevance_score": 0.05},
    {"index": 2, "relevance_score": 0.72},
    """
    scores = [0.0]*len(documents) #先创建一个和documents长度一样的分数列表,比如[0.0, 0.0, 0.0, 0.0]
    for item in results:
        index = item.get("index")
        score = item.get("relevance_score")
        scores[int(index)] = float(score)
    return scores
