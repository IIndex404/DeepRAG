import logging
from pathlib import Path
from collections.abc import Sequence

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from config.embedding_config import embedding_config
from processor.import_processor.base import setup_logging
setup_logging()
_bge_m3_ef = None #模型单例对象，避免重复初始化
#get_bge_m3_ef()=========》BGE-M3 向量模型=========》返回一个BGE-M3 模型
#generate_embeddings(texts)=========》把文本转成 dense 和 sparse 两种向量=========》返回稠密向量和稀疏向量，这俩都是字典嵌套再字典里
# 这个返回的是{
#     "dense": [
#         [0.1, 0.2, 0.3],
#         [0.4, 0.5, 0.6]
#     ],
#
#     "sparse": [
#         {2: 0.8, 5: 0.3},
#         {1: 0.6, 7: 0.9}
#     ]
# }
def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return:初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is None:
        # 优先使用本地模型；本地目录尚未下载时，回退到模型名称并自动下载。
        local_model_path = embedding_config.bge_m3_path
        if local_model_path and Path(local_model_path).exists():
            model_name = local_model_path
        else:
            model_name = embedding_config.bge_m3
            if not model_name:
                raise RuntimeError("未配置 BGE_M3_PATH 或 BGE_M3，无法加载嵌入模型")
            logging.warning(
                "BGE-M3 本地目录不存在：%s，将使用模型名称 %s 加载",
                local_model_path,
                model_name,
            )

        device = embedding_config.bge_device or "cpu"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PyTorch，无法加载 BGE-M3；请先同步项目依赖"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            logging.warning("CUDA 当前不可用，BGE-M3 将回退到 CPU")
            device = "cpu"

        # CPU 不支持这里需要的 FP16 推理，只有 CUDA 模式才启用半精度。
        use_fp16 = embedding_config.bge_fp16 and device.startswith("cuda")
        logging.info("正在加载 BGE-M3：model=%s, device=%s", model_name, device)
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
        )
    return _bge_m3_ef


def generate_embeddings(texts: str | Sequence[str]):
    """
    为文本生成向量嵌入
    :param texts: 单条文本或文本列表；内部统一转换成 List[str]
    :return: 包含dense和sparse向量的字典
    """
    if isinstance(texts, str):
        normalized_texts = [texts]
    elif isinstance(texts, Sequence):
        normalized_texts = list(texts)
    else:
        raise TypeError("texts 必须是 str 或字符串列表")

    if not normalized_texts:
        raise ValueError("texts 不能为空")
    if not all(isinstance(text, str) and text.strip() for text in normalized_texts):
        raise ValueError("texts 中每一项都必须是非空字符串")

    model = get_bge_m3_ef()
    embeddings = model.encode_documents(normalized_texts)
    processed_sparse = [] #3.创建一个processed_sparsea列表用来装后面生成的sparse稀疏向量
    for i in range(len(normalized_texts)): #一个一个文本的处理
        #这里加一个系数矩阵属性：indices = 哪些位置有值，data=这些位置具体是多少，indptr=每条文本的数据从哪里开始、到哪里结束
        # 从所有文本的稀疏向量位置里，切出第 i 条文本自己的那一段，再转成普通列表
        sparse_indices = embeddings["sparse"].indices[
            embeddings["sparse"].indptr[i]:
            embeddings["sparse"].indptr[i + 1]
        ].tolist()
        sparse_data = embeddings["sparse"].data[
            embeddings["sparse"].indptr[i]:
            embeddings["sparse"].indptr[i + 1]
        ].tolist()
        #把“位置”和“数值”一一对应起来
        # 比如：indices = [2, 8, 15] data = [0.3, 0.9, 0.5] 最后变成：{2: 0.3, 8: 0.9, 15: 0.5}
        sparse_dict = { # 把“有值的位置”和“对应的数值”一一配对，整理成字典
            k:v
            for k, v in zip(sparse_indices, sparse_data)
        }
        processed_sparse.append(sparse_dict)
    return {
        #dense原本可能是numpy数组，这里转成普通python列表
        "dense": [emb.tolist()
                  for emb in embeddings["dense"]],
        #sparse用的是我们上面已经处理好的字典列表
        "sparse":processed_sparse
    }
"""
最后输出的实例如此
{
    "dense": [
        [0.1, 0.2, 0.3, ...],   # 第1条文本的稠密向量
        [0.4, 0.5, 0.6, ...]    # 第2条文本的稠密向量
    ],

    "sparse": [
        {2: 0.8, 10: 1.2},      # 第1条文本的稀疏向量
        {3: 0.5, 20: 0.9}       # 第2条文本的稀疏向量
    ]
}
"""
