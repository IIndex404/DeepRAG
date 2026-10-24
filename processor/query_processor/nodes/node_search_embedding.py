"""
这个节点功能是
先把用户的问题转成向量
然后去Milvus里搜索
找到相似的chunks就可以返回Top—K了
"""
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import create_hybrid_search_requests, get_milvus_client,hybrid_search
class NodeSearchEmbedding(NodeBase):
    name:str = "node_search_embedding"
    def process(self,state:QueryGraphState)->QueryGraphState:
        """
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        try:
            # 1、先从state里面拿前面节点已经处理好的问题和商品名
            query = state.get("rewritten_query") #拿大模型已经重写完整的用户问题
            item_names = state.get("item_names") #拿上一个商品确认节点已经确认好的商品名列表
            # 2、把用户问题转成稠密向量和稀疏向量
            if not isinstance(query, str) or not query.strip():
                raise ValueError("rewritten_query 不能为空")
            embeddings = generate_embeddings([query]) #统一按文本列表调用BGE-M3
            dense_vec = embeddings.get("dense")[0] #拿当前问题生成的第一个稠密向量，主要负责语义相似
            sparse_vec = embeddings.get("sparse")[0] #拿当前问题生成的第一个稀疏向量，主要负责关键词匹配
            # 3、拿到Milvus里面专门保存知识切片的集合名字
            collection_name = milvus_config.chunks_collection #后面就去这个chunks集合里面搜索真正的知识片段
            # 4、准备商品名过滤条件
            expr = None  # 先默认不加任何过滤条件
            if item_names: #如果前面已经确认出了商品名
                import json
                expr = f"item_name in {json.dumps(item_names, ensure_ascii=False)}" #Milvus过滤表达式使用JSON列表
                logger.info(f"过滤条件:{expr}") #把这次真正使用的过滤条件打印出来
            else:
                logger.info("未指定商品名过滤，将全库检索") #如果没有商品名，那就不限制商品范围，直接搜整个知识库
            # 5、把稠密向量+稀疏向量包装成Milvus能够执行的两个搜索请求
            reqs = create_hybrid_search_requests(
                dense_vector = dense_vec,
                sparse_vector = sparse_vec,
                expr=expr,
                limit=10,
            )
            # 6、开始真正执行Milvus混合向量搜索
            logger.info("开始执行 Milvus 混合检索...")
            client = get_milvus_client() #先拿Milvus客户端，也就是先连接上Milvus
            res = hybrid_search(
                  client = client,
                  collection_name = collection_name,
                  reqs = reqs,
                  ranker_weights=(0.8, 0.2),
                  limit=10,
                  output_fields = ["chunk_id","content","item_name"]#返回知识切片自己的id,返回知识切片正文,返回这个知识切片属于哪个商品
            )
            # 7、把Milvus搜出来的结果放进embedding_chunks返回给后面的节点
            return {"embedding_chunks": res[0] if res else []} #如果搜到结果就拿第一组结果，没有就返回空列表
        except Exception as e:
            logger.exception(f"向量搜索失败:{e}")
            raise
if __name__ == "__main__": #只有直接运行这个py文件的时候，才执行下面的单元测试

    init_state = { #模拟上一个节点已经处理完成以后传过来的state
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？", #已经被大模型重写完整的问题
        "item_names": [ #已经确认好的标准商品名
            "BrotherHAK180烫金机",
            "BrotherHAK-180烫金机"
        ]
    }

    node_search_embedding = NodeSearchEmbedding() #实例化一个普通向量搜索节点
    result = node_search_embedding(init_state) #把模拟state交给节点完整跑一次

    logger.info(
        serialize_json(
            result,
            indent=4
        )
    ) #把最终搜索结果转成好看的JSON格式打印出来
