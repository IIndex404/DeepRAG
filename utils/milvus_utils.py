import os
from pymilvus import MilvusClient
from pymilvus.client.types import MetricType

from config.milvus_config import milvus_config
from pymilvus import AnnSearchRequest,WeightedRanker
from tool.logger import logger
_milvus_client = None # Milvus客户端单例对象，避免每次调用都重新连接一次Milvus
#======获取客户端的一个方法======
def get_milvus_client():
    """
    获取全局唯一的Milvus客户端对象
    :return: Milvus客户端对象
    """
    global _milvus_client  #如果改 _milvus_client，我要改的是外面那个。
    if _milvus_client is not None:
        return _milvus_client
    _milvus_client = MilvusClient(
        uri=milvus_config.milvus_url,
    )
    return _milvus_client
#======把字符串里的特殊字符处理一下，避免后面拼Milvus查询条件时报错的一个方法======
def escape_milvus_string(value:str) -> str : # 把字符串里的特殊字符处理一下，避免后面拼Milvus查询条件时报错
    value = value.replace(
        "\\", "\\\\"
    ).replace(
        '"', '\\"'
    ).replace(
        "'", "\\'"
    )
    return value
#======构建混合搜索请求的一个方法======
def create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        dense_params=None,
        sparse_params=None,
        expr=None,
        limit=5,):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector:文本生成的稀疏向量
    :param dense_params:稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params:稀疏向量搜索参数，默认使用内积相似度
    :param expr:搜索过滤表达式，用于精准筛选数据
    :param limit:单向量搜索返回结果数量，默认5
    :return:搜索请求列表，包含[dense_req, sparse_req]
    """
    if dense_params is None:
        dense_params = {"metric_type":"COSINE"} #如果没有稠密向量搜索参数，默认使用余弦相似度
    if sparse_params is None:
        sparse_params = {"metric_type":"IP"} #如果没有稀疏向量搜索参数，默认使用内积相似度
    #构建稠密向量搜索请求
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit,)
    #构建稀疏向量搜索请求
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit,)
    return [dense_req, sparse_req]
# ============ 执行混合搜索的方法 ============
def hybrid_search(
        client,
        collection_name,
        reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=False,
        limit=5,
        output_fields = None,
        search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 稠密/稀疏向量权重
    :param norm_score: 是否先归一化评分再融合
    :param limit: 最终返回结果数量
    :param output_fields: 需要返回的字段
    :param search_params: 搜索参数
    :return: 混合搜索结果，失败返回None
    """
    try:
        rerank = WeightedRanker(
            ranker_weights[0],
            ranker_weights[1],
            norm_score=norm_score
        )
        # 如果输出是空的话，默认返回商品名称
        if output_fields is None:
            output_fields = ["item_name"]
        # 执行混合搜索
        res = client.hybrid_search(
            collection_name = collection_name,
            reqs = reqs,
            ranker = rerank,
            limit = limit,
            output_fields = output_fields,
            )
        logger.info(f"Milvus混合搜索完成，集合:[{collection_name}]"
                    f"共检索到{len(res[0])}结果")
        return res
        """
        res = 
        [[
            {"id": 3, "distance": 0.95, "entity": {"item_name": "苹果iPhone15"}},
            {"id": 7, "distance": 0.82, "entity": {"item_name": "苹果iPhone14"}},
        ]]  #输出是id，对应的分数和他们对应的实体
        """
    except Exception as e:
        logger.error(
            f"Milvus混合搜索失败，集合:[{collection_name}]：{str(e)}",
            exc_info = True #把报错的完整堆栈也打印出来
        )
        raise
