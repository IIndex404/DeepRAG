"""
用户问题转为向量，去Milvus找相似chunk
hyde检索：先让大模型生成一个假答案，拿假答案去搜索，提高检索准确度
最后把普通检索、HyDE检索和联网搜索找到的结果放到一起做RRF排序,Rerank再精排一次
最终分数最高的给大模型回答
state = {
    # ===== 会话标识 =====
    "session_id": "abc123",              # 会话ID
    "message_id": "msg001",              # 消息ID（可选，可以先不管）

    # ===== 用户问题 =====
    "original_query": "怎么调温度？",      # 用户原始问题
    "rewritten_query": "关于HAK180烫金机，如何调节转印温度？",  # 改写后的问题

    # ===== 商品名 =====
    "item_names": ["HAK180烫金机"],       # 确认好的商品名列表

    # ===== 历史对话 =====
    "history": [
        {"role": "user", "text": "你好"},
        {"role": "assistant", "text": "你好，请问有什么可以帮你？"},
    ],

    # ===== 三路检索结果 =====
    "embedding_chunks": [                 # 向量搜索的结果
        {"entity": {"chunk_id": "c1", "content": "温度调到110度"}},
        {"entity": {"chunk_id": "c2", "content": "按住预热键3秒"}},
    ],
    "hyde_embedding_chunks": [            # HyDE 搜索的结果
        {"entity": {"chunk_id": "c1", "content": "温度调到110度"}},
        {"entity": {"chunk_id": "c3", "content": "转印时间设为5秒"}},
    ],
    "hyde_doc": "HAK180的转印温度调节方法如下...",  # HyDE 生成的假设答案
    "web_search_docs": [                 # 联网搜索的结果
        {"title": "HAK180使用说明", "url": "https://example.com", "snippet": "温度调节方法..."},
    ],

    # ===== 融合 + 重排 =====
    "rrf_chunks": [                       # RRF 融合后的结果
        {"chunk_id": "c1", "content": "温度调到110度"},
        {"chunk_id": "c2", "content": "按住预热键3秒"},
        {"chunk_id": "c3", "content": "转印时间设为5秒"},
    ],
    "reranked_docs": [                    # Rerank 精排后的结果
        {"chunk_id": "c1", "content": "温度调到110度", "score": 0.95, "source": "local"},
        {"chunk_id": "c3", "content": "转印时间设为5秒", "score": 0.72, "source": "local"},
    ],

    # ===== 最终生成 =====
    "prompt": "你是一个智能助手...【参考内容】...【用户问题】...",  # 拼好的提示词
    "answer": "HAK180的转印温度调节方法：先接通电源...",          # 最终答案

    # ===== 控制标记 =====
    "is_stream": False,                   # 是否流式输出
}
"""
from typing import Tuple,Dict,Any,List
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
class NodeRrf(NodeBase):
    name:str = "node_rrf"
    def process(self, state:QueryGraphState)->QueryGraphState:
        """
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 1.拿普通向量搜索返回的结果
        # embedding_chunks里面每一条外面还有一层entity
        # 所以这里把真正的chunk内容entity拿出来
        embedding_search_list = [
            {
                **doc.get("entity"),
                "source": "local",
                "url": None,
            }
            for doc in (state.get('embedding_chunks') or [])
            if isinstance(doc,dict) and isinstance(doc.get("entity"), dict)
        ] #这里会得到一个{"chunk_id": "c1", "content": "温度调到110度"}里面有字典是chunk_id和content
        # 2.拿HyDE搜索返回的结果
        # 同样把每条结果里面真正的entity拿出来
        hyde_embedding_chunks = [
            {
                **doc.get("entity"),
                "source": "local",
                "url": None,
            }
            for doc in (state.get('hyde_embedding_chunks') or [])
            if isinstance(doc,dict) and isinstance(doc.get("entity"), dict)
        ] #这里会得到一个{"chunk_id": "c1", "content": "温度调到110度"}里面有字典是chunk_id和content
        # 3.把联网结果也整理成和本地切片一致的格式
        web_search_list = [
            {
                "chunk_id": None,
                "content": doc.get("snippet") or "",
                "title": doc.get("title"),
                "url": doc.get("url"),
                "source": "web",
            }
            for doc in (state.get("web_search_docs") or [])
            if isinstance(doc, dict) and (doc.get("snippet") or "").strip()
        ]
        # 4.普通检索、HyDE检索和联网搜索使用相同权重进入RRF
        rrf_inputs = [
            (embedding_search_list,1.0),  # 元组1：向量搜索结果 + 权重
            (hyde_embedding_chunks,1.0),  # 元组2：HyDE搜索结果 + 权重
            (web_search_list,1.0),         # 元组3：联网搜索结果 + 权重
        ]

        # 5.调用下面自己写的_rrf_merge
        # 给所有chunk重新计算RRF分数并重新排序
        #rrf_inputs = [([{"chunk_id": "content1"}, {"chunk_id": "content2"}], 1.0),
        #              ([{"chunk_id": "content1"}, {"chunk_id": "content2"}], 1.0),]
        rrf_merge_results = self._rrf_merge(rrf_inputs) #得到计算rrf分数后的结果
        # 6._rrf_merge返回的是：
        # [(文档1, 分数1), (文档2, 分数2)...]
        # 这里我们只要文档，不要RRF分数
        rrf_chunks = [doc for doc , _ in rrf_merge_results ] #_ 就是占位符表示我不要这个东西的意思
        # 7.把最终融合排序好的结果放回state
        # 后面的Rerank节点继续使用
        state["rrf_chunks"] = rrf_chunks
        logger.info(
            "RRF三路融合完成：普通检索%d条，HyDE%d条，联网%d条，去重后%d条",
            len(embedding_search_list),
            len(hyde_embedding_chunks),
            len(web_search_list),
            len(rrf_chunks),
        )
        # 8.返回更新后的state
        return state
    def _rrf_merge(self, rrf_inputs,k:int=60,max_results:int=None) -> list[Tuple[Dict[str,Any],float]]:
        """
        :param rrf_inputs:[(第一路搜索结果, 权重),(第二路搜索结果, 权重)]
        :param k:  RRF平滑常数，默认60
        :param max_results: 最后最多要多少条结果,None就是全部都要
        :return: [(文档, RRF分数), ...]
        """
        chunk_scores = {} #切片分数
        chunk_data = {} #切片数据
        for rrf_input,weight in rrf_inputs:
            for rank,doc in enumerate(rrf_input,start=1):
                # 本地切片用chunk_id去重，联网结果用URL去重，没有时再用正文兜底。
                document_id = doc.get("chunk_id") or doc.get("url")
                if not document_id:
                    document_id = f"content:{doc.get('content', '')}"
                chunk_scores[document_id] = chunk_scores.get(document_id,0) + weight/(k+rank)
                chunk_data.setdefault(document_id,doc)
        # 这一步就是把文本和分数合起来得到未排序结果
        unsorted_results = [(chunk_data[cid],score) for cid,score in chunk_scores.items()]
        # 按每一条结果的第二个元素，也就是RRF分数排序,key=lambda x: x[1] = 按第二个排,,reverse=True = 从大到小
        sorted_results = sorted(unsorted_results, key=lambda x:x[1], reverse=True)
        # 如果指定了max_results，就只拿前max_results条,没指定就把全部结果返回
        return (sorted_results[:max_results]) if max_results else sorted_results
if __name__ == "__main__":

    #模拟前面两路搜索节点已经搜出来的结果
    mock_state = {

        #普通向量搜索结果
        "embedding_chunks": [
            {
                "entity": {
                    "chunk_id": "chunk_1",
                    "content": "向量搜索结果#1"
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_2",
                    "content": "向量搜索结果#2"
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_3",
                    "content": "向量搜索结果#3"
                }
            },
        ],

        #HyDE搜索结果
        "hyde_embedding_chunks": [
            {
                "entity": {
                    "chunk_id": "chunk_1",
                    "content": "HyDE搜索结果#1"
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_4",
                    "content": "HyDE搜索结果#2"
                }
            },
            {
                "entity": {
                    "chunk_id": "chunk_2",
                    "content": "HyDE搜索结果#3"
                }
            },
        ]
    }

    #创建RRF节点
    node_rrf = NodeRrf()

    #把模拟state交给RRF节点
    result = node_rrf(mock_state)

    #把最后结果打印出来
    logger.info(
        serialize_json(
            result,
            indent=4
        )
    )
