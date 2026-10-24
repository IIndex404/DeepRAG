"""
大模型根据用户的问题生成一个假设答案，
再用“用户问题 + 假设答案”向量检索 Milvus 中的知识切片，
结果保存到 hyde_embedding_chunks。
"""
import json

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompts.hyde_prompt import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.llm_utils import get_llm_client
from utils.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)


class NodeSearchEmbeddingHyde(NodeBase):
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """生成 HyDE 文档，并检索 Milvus 中的知识切片。"""
        rewritten_query = (
            state.get("rewritten_query")
            or state.get("original_query")
            or ""
        ).strip()
        item_names = state.get("item_names") or []

        try:
            # 没有目标集合时，无需调用大模型或加载 BGE-M3。
            client = get_milvus_client()
            collection_name = milvus_config.chunks_collection
            if not collection_name or not client.has_collection(collection_name):
                logger.warning(
                    f"Milvus集合[{collection_name}]不存在，请先导入知识库；"
                    "本次HyDE检索返回空列表"
                )
                return {"hyde_embedding_chunks": [], "hyde_doc": ""}

            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)
            chunks = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names,
                client=client,
            )
            return {
                "hyde_embedding_chunks": chunks,
                "hyde_doc": hyde_doc,
            }
        except Exception as e:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        logger.info("步骤1: 开始生成假设性文档")
        if not rewritten_query:
            logger.warning("步骤1: 用户问题为空，跳过假设文档生成")
            return ""

        try:
            llm = get_llm_client()
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            response = llm.invoke(hyde_prompt)
            hyde_doc = str(response.content or "").strip()
            logger.info(f"步骤1: 假设文档内容长度: {len(hyde_doc)}")

            if not hyde_doc:
                logger.warning("步骤1: 假设文档为空，回退使用原问题")
                return rewritten_query
            return hyde_doc
        except Exception as e:
            logger.exception(f"步骤1: 生成假设文档失败: {e}")
            raise

    def _step_2_search_embedding_hyde(
        self,
        rewritten_query: str,
        hyde_doc: str,
        item_names=None,
        client=None,
    ) -> list:
        try:
            collection_name = milvus_config.chunks_collection
            client = client or get_milvus_client()
            if not collection_name or not client.has_collection(collection_name):
                logger.warning(
                    f"Milvus集合[{collection_name}]不存在，请先导入知识库；"
                    "本次HyDE检索返回空列表"
                )
                return []

            combined_text = f"{rewritten_query or ''} {hyde_doc or ''}".strip()
            if not combined_text:
                logger.warning("步骤2: 检索文本为空，本次HyDE检索返回空列表")
                return []

            embeddings = generate_embeddings([combined_text])
            dense_vectors = embeddings.get("dense") or []
            sparse_vectors = embeddings.get("sparse") or []
            if not dense_vectors or not sparse_vectors:
                raise ValueError("BGE-M3未返回完整的dense/sparse向量")

            expr = None
            normalized_item_names = [
                str(item_name).strip()
                for item_name in (item_names or [])
                if str(item_name).strip()
            ]
            if normalized_item_names:
                item_names_json = json.dumps(
                    normalized_item_names,
                    ensure_ascii=False,
                )
                expr = f"item_name in {item_names_json}"
                logger.info(f"步骤2: 过滤条件: {expr}")
            else:
                logger.info("步骤2: 未指定商品名过滤，将全库检索")

            reqs = create_hybrid_search_requests(
                dense_vector=dense_vectors[0],
                sparse_vector=sparse_vectors[0],
                expr=expr,
                limit=10,
            )
            logger.info("步骤2: 开始执行 Milvus 混合检索...")
            res = hybrid_search(
                client=client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                limit=10,
                output_fields=["chunk_id", "content", "item_name"],
            )
            return res[0] if res else []
        except Exception as e:
            logger.exception(f"步骤2: 检索过程发生异常: {e}")
            raise


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": [
            "BrotherHAK180烫金机",
            "BrotherHAK-180烫金机",
        ],
    }

    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(serialize_json(result, indent=4))