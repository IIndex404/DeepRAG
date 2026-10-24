"""
Rerank精排一次
"""
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from typing import List,Dict,Any
from utils.reranker_http_utils import rerank_documents
RERANK_MAX_TOPK:int = 10 # 最多保留前 10 条
RERANK_MIN_TOPK:int = 3 # 最少保留前 3 条
RERANK_GAP_ABS:float = 0.5 # 断崖绝对阈值
RERANK_GAP_RATIO:float = 0.25 # 断崖相对阈值
class NodeRerank(NodeBase):
    name:str = "node_rerank"
    def process(self, state:QueryGraphState)->QueryGraphState:
        """
        Rerank精排主流程
        1. 合并RRF结果和联网搜索结果
        2. 调Rerank模型重新打分排序
        3. 根据分数断崖决定保留多少条
        4. 最终结果放进reranked_docs
        state =
        {"rewritten_query": "HAK180烫金机怎么调温度？",
        # 本地 RRF 融合完的结果
        "rrf_chunks": [
        {"chunk_id": "c1", "content": "温度调到110度", "title": "温度设置"},
        {"chunk_id": "c2", "content": "按住预热键3秒", "title": "预热操作"},
        ],
        # 联网搜索的结果
        "web_search_docs": [
        {"title": "HAK180手册", "url": "http://xxx", "snippet": "温度调节方法..."},],}
        """
        # 1.把本地RRF结果和联网搜索结果合并到一个列表
        merged_multi_docs:List[Dict[str, Any]] =  self._step_1_merge_multi_source_docs(state)
        # 2.把这些文档交给Rerank模型
        # 让模型重新打相关性分数，然后从高到低排序
        reranked_docs:List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state,merged_multi_docs)
        # 3.检查分数有没有突然大幅下降,根据断崖位置决定最后保留几条
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)
        # 4.把最终筛选出来的文档保存回全局state
        state["reranked_docs"] = cutoff_docs
        # 5.返回state给下一个节点
        return state
    def _step_1_merge_multi_source_docs(self,state: QueryGraphState)->List[Dict[str, Any]]:
        """
        RRF已经完成三路融合，这里只把融合结果整理成统一格式。
        """
        final_docs = []  # 最后所有文档都放到这个列表里
        # 1.拿出三路RRF融合结果
        for rrf_doc in (state.get("rrf_chunks") or []):
            format_rrf_doc = {
                "content":rrf_doc.get("content"),
                "title":rrf_doc.get("title"),
                "chunk_id":rrf_doc.get("chunk_id"),
                "url":rrf_doc.get("url"),
                "source":rrf_doc.get("source") or "local"
            }
            if (format_rrf_doc["content"] or "").strip():
                final_docs.append(format_rrf_doc)
        return final_docs
    def _step_2_rerank_merged_docs(self,state:QueryGraphState,merged_multi_docs:List[Dict[str, Any]])->List[Dict[str, Any]]:
        # 1.拿到用户经过改写后的问题
        user_query = state.get("rewritten_query")
        if not user_query:
            raise ValueError("Rerank缺少rewritten_query")
        if not merged_multi_docs:
            raise ValueError("Rerank没有收到任何检索结果")
        # 2.只把所有文档的content正文拿出来
        contents = [doc.get("content") for doc in merged_multi_docs]
        # 3.把用户问题 + 所有文档一起交给Rerank模型
        rerank_scores = rerank_documents(user_query,contents)
        if len(rerank_scores) != len(merged_multi_docs):
            raise ValueError(
                f"Rerank返回数量不一致：文档{len(merged_multi_docs)}条，分数{len(rerank_scores)}条"
            )
        # 4.把Rerank给出的分数,塞回对应的原文档里面
        scored_docs = [{**doc,"score": score} for doc,score in zip(merged_multi_docs,rerank_scores)]
        # 5.按照score从大到小重新排序
        return sorted(scored_docs, key=lambda doc: doc["score"], reverse=True)
    def _step_3_cliff_cutoff(self,reranked_docs:List[Dict[str, Any]])->List[Dict[str, Any]]:
        #断崖检测,看相邻两篇文档的分数有没有突然大幅下降。例如：0.88,0.12,0.08,0.88 → 0.12突然掉很多,那么就在这里截断，最终只保留前3条。
        """
        reranked_docs = [
    {
        "content": "HAK180烫金机的转印温度调节方法：接通电源后，按住设置键3秒，温度调到110度。",
        "title": "温度设置",
        "chunk_id": "c1",
        "url": None,
        "source": "local",
        "score": 0.95
    },
    {
        "content": "烫金机的日常保养：定期清理加热板，避免灰尘堆积影响温度。",
        "title": "日常保养",
        "chunk_id": "c3",
        "url": None,
        "source": "local",
        "score": 0.72
    },
    {
        "content": "温度调节方法：在操作面板上找到温度按钮，按上下键调节。",
        "title": "HAK180手册",
        "url": "http://xxx",
        "snippet": "...",
        "source": "web",
        "score": 0.68
    }
]

        """
        if not reranked_docs:
            return []
        upper_bound = min(RERANK_MAX_TOPK,len(reranked_docs))  #最多检查前10条
        lower_bound = min(RERANK_MIN_TOPK,len(reranked_docs))  #最少保留3条
        cutoff_pos = upper_bound  #先把截断位置设成"最多条数"，也就是默认全留。
        for idx in range(lower_bound-1,upper_bound-1):
            current_score = reranked_docs[idx].get("score") #当前文档的分数
            next_score = reranked_docs[idx+1].get("score") #下一篇文档的分数
            if current_score is None or next_score is None:  # 如果没有分数,就没办法做断崖判断
                continue
            abs_gap = current_score - next_score # 计算绝对下降多少
            rel_gap = abs_gap/(abs(current_score)+ 1e-6) # 计算相对下降比例
            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO: # 判断是不是出现断崖,满足其中一个条件就算断崖
                cutoff_pos = idx + 1 #"索引2这里是断崖，那就保留到索引2为止。索引2是第 3 条，所以保留3条。+1就是把索引翻译成条数。
                logger.debug(
                    f"断崖检测: "
                    f"位置 {idx + 1}, "
                    f"abs_gap={abs_gap:.4f}, "
                    f"rel_gap={rel_gap:.4f}"
                )
                break
        return reranked_docs[:cutoff_pos]
if __name__ == "__main__":

    # 模拟前面节点传进来的state
    mock_state = {
        "rewritten_query": "HAK180烫金机怎么调温度？",

        # 本地RRF融合后的结果
        "rrf_chunks": [
            {
                "chunk_id": "c1",
                "content": "HAK180烫金机需要先按住设置键3秒，然后把温度调整到110度。",
                "title": "温度设置"
            },
            {
                "chunk_id": "c2",
                "content": "机器使用完成以后，应及时关闭电源并清理加热板。",
                "title": "设备保养"
            },
            {
                "chunk_id": "c3",
                "content": "烫金纸需要根据材料类型选择不同规格。",
                "title": "烫金纸选择"
            }
        ],

        # 联网搜索结果
        "web_search_docs": [
            {
                "title": "HAK180使用说明",
                "url": "http://example.com/hak180",
                "snippet": "HAK180温度可以通过操作面板上的上下按键进行调节。"
            },
            {
                "title": "烫金机维修知识",
                "url": "http://example.com/repair",
                "snippet": "机器出现异响时需要检查传动结构和电机。"
            }
        ]
    }

    # 创建Rerank节点
    node_rerank = NodeRerank()

    # 执行整个Rerank节点
    result = node_rerank.process(mock_state)

    print("\n========== Rerank最终结果 ==========")

    # 看最终留下来的文档
    for index, doc in enumerate(
            result["reranked_docs"],
            start=1
    ):
        print(f"\n第{index}条")

        print(
            f"标题：{doc.get('title')}"
        )

        print(
            f"来源：{doc.get('source')}"
        )

        print(
            f"分数：{doc.get('score')}"
        )

        print(
            f"内容：{doc.get('content')}"
        )
