"""把带有 BGE-M3 双向量的文档切片写入 Milvus。"""

from typing import Any, Dict, List

from pymilvus import DataType, MilvusClient

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import MilvusError, StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import escape_milvus_string, get_milvus_client


class NodeImportMilvus(BaseNode):
    name = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks, vector_dimension = self._step_1_check_input(state)
        client = self._step_2_prepare_collection(vector_dimension)
        self._step_3_clean_old_data(client, chunks)
        state["chunks"] = self._step_4_insert_data(client, chunks)
        return state

    def _step_1_check_input(
        self,
        state: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise StateFieldError(
                field_name="chunks",
                message="chunks 必须是非空列表",
                expected_type=list,
            )

        for index, chunk in enumerate(chunks):
            if not chunk.get("content"):
                raise StateFieldError(
                    field_name="chunks",
                    message=f"第 {index + 1} 个 chunk 缺少 content",
                )
            if not chunk.get("dense_vector"):
                raise StateFieldError(
                    field_name="chunks",
                    message=f"第 {index + 1} 个 chunk 缺少 dense_vector",
                )
            if not isinstance(chunk.get("sparse_vector"), dict):
                raise StateFieldError(
                    field_name="chunks",
                    message=f"第 {index + 1} 个 chunk 缺少 sparse_vector",
                )

        vector_dimension = len(chunks[0]["dense_vector"])
        if vector_dimension <= 0:
            raise StateFieldError(
                field_name="dense_vector",
                message="稠密向量维度必须大于 0",
            )
        if any(len(chunk["dense_vector"]) != vector_dimension for chunk in chunks):
            raise StateFieldError(
                field_name="dense_vector",
                message="所有 chunk 的稠密向量维度必须一致",
            )
        return chunks, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension: int) -> MilvusClient:
        client = get_milvus_client()
        collection_name = milvus_config.chunks_collection
        if not collection_name:
            raise MilvusError("未配置 CHUNKS_COLLECTION")

        if not client.has_collection(collection_name):
            self._create_chunks_collection(
                collection_name,
                client,
                vector_dimension,
            )
            self.logger.info("已创建 Milvus 集合：%s", collection_name)
        return client

    def _create_chunks_collection(
        self,
        collection_name: str,
        client: MilvusClient,
        vector_dimension: int,
    ) -> None:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="parent_title",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(field_name="part", datatype=DataType.INT32)
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=vector_dimension,
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _step_3_clean_old_data(
        self,
        client: MilvusClient,
        chunks: List[Dict[str, Any]],
    ) -> None:
        file_title = str(chunks[0].get("file_title") or "").strip()
        if not file_title:
            raise StateFieldError(
                field_name="file_title",
                message="chunk 中缺少 file_title，无法执行幂等导入",
            )

        safe_file_title = escape_milvus_string(file_title)
        try:
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f'file_title == "{safe_file_title}"',
            )
        except Exception as exc:
            raise MilvusError(f"清理旧切片失败：{exc}") from exc

    def _step_4_insert_data(
        self,
        client: MilvusClient,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        fields = (
            "content",
            "title",
            "parent_title",
            "part",
            "file_title",
            "item_name",
            "sparse_vector",
            "dense_vector",
        )
        data_to_insert = []
        for chunk in chunks:
            row = {field: chunk.get(field) for field in fields}
            row["title"] = str(row.get("title") or "")
            row["parent_title"] = str(
                row.get("parent_title") or row["title"]
            )
            row["part"] = int(row.get("part") or 0)
            row["file_title"] = str(row.get("file_title") or "")
            row["item_name"] = str(row.get("item_name") or "")
            data_to_insert.append(row)

        try:
            result = client.insert(
                collection_name=milvus_config.chunks_collection,
                data=data_to_insert,
            )
            inserted_ids = result.get("ids") or []
            if len(inserted_ids) != len(chunks):
                raise MilvusError(
                    f"Milvus 返回 {len(inserted_ids)} 个主键，"
                    f"但本次有 {len(chunks)} 个切片"
                )
            for chunk, chunk_id in zip(chunks, inserted_ids):
                chunk["chunk_id"] = chunk_id
            client.flush(milvus_config.chunks_collection)
            client.load_collection(milvus_config.chunks_collection)
            self.logger.info(
                "已写入 Milvus 集合 %s，共 %s 条切片",
                milvus_config.chunks_collection,
                len(chunks),
            )
            return chunks
        except Exception as exc:
            if isinstance(exc, MilvusError):
                raise
            raise MilvusError(f"切片写入 Milvus 失败：{exc}") from exc
