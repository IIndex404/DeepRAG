from typing import TypedDict,List
class QueryGraphState(TypedDict):
    """
    这里规定整个问答查询流程的state结构
    后面的检索、排序、重写、生成等节点都从这个state里拿数据，
    处理完再把新的结果写回state
    """
    session_id: str #会话id
    message_id: str #消息id
    original_query: str #用户原始问题
    #检索过程中的中间数据
    embedding_chunks:list #普通向量检索回来的切片
    hyde_embedding_chunks: list #已向量化的假设性问题切片
    web_search_docs:list #网络搜索回来的文档
    #排序过程中的数据
    rrf_chunks:list #RRF融合排序后的切片
    reranked_docs:list #重排序后的最终Top-K文档
    #生成过程中的数据
    prompt: str #组装好的Prompt
    answer: str #最终生成的答案
    item_names: List[str] #提取出的商品名称
    rewritten_query:str #重写后的问题
    history:list #历史对话记录
    is_stream:bool #是否流式输出


