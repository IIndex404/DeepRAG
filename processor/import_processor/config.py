""" 这个文件专门用来“统一放项目里的各种设置"""
from dataclasses import dataclass,field
from typing import Set,Optional
import os
from dotenv import load_dotenv
load_dotenv()
@dataclass
class ImportConfig:
    """把导入流程需要的所有设置统一放在这个类里面"""
    # ==================== 文档怎么处理 ====================
    # 一个文本块最多放 2000 个字符
    max_content_length:int = 2000
    # 处理图片时，最多看看图片前后 200 个字符
    img_content_length: int = 200
    # 如果一段文字太短，小于 500，就尽量和别的内容合并
    min_content_length: int = 500
    # 切文本时，前后保留 1 句话重复一下，防止上下文断掉
    overlap_sentences:int = 1
    # 判断商品名时，先拿前 3 个文本块来分析
    item_name_chunk_k:int = 3
    #判断商品名字的时候，最多给大模型看2500个字符
    item_name_chunk_size: int = 2500
    # 这些图片格式都允许处理
    image_extensions:Set[str] = field(
        default_factory=lambda:{
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp"
        })
    # ==================== LLM 配置 ====================
    # 大模型 API 地址
    openai_api_base:str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE","")
    )
    # 大模型 API Key
    openai_api_key:str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY","")
    )
    # 视觉模型
    vl_model:str = field(
        default_factory=lambda: os.getenv("VL_MODEL","")
    )
    # 商品名称识别模型
    item_model:str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL","")
    )
    # 默认大模型
    default_model:str = field(
        default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL","")
    )
    # ==================== Milvus 配置 ====================
    # 从 .env 里拿 Milvus 的连接地址
    milvus_url:str = field(
        default_factory=lambda: os.getenv("MILVUS_URL","")
    )
    # 存“文档切片”的 Milvus 表叫什么
    chunks_collection:str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION","")
    )
    # 存“商品名称”的 Milvus 表叫什么
    item_name_collection:str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION","")
    )
    # ==================== MinIO 相关设置 ====================
    # 从 .env 里拿 MinIO 的地址
    minio_endpoint:str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT","")
    )
    # 从 .env 里拿 MinIO 的账号
    minio_access_key:str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY","")
    )
    # 从 .env 里拿 MinIO 的密码
    minio_secret_key:str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY","")
    )
    # 文件要存进 MinIO 里的哪个桶
    minio_bucket:str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME","")
    )
    # False 就表示我们现在本地用的是 http，不是 https
    minio_secure:bool = False
    # ==================== 向量相关设置 ====================
    #向量有多少维度，从env读取，如果没有，就默认1024
    embedding_dim:int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM","1024"))
    )
    #做向量时候，一次处理8条数据
    embedding_batch_size:int = 8
    # ==================== API 调用次数限制 ====================
    #图片总结接口一分钟最多调用15次，防止请求太快被限流
    requests_per_minute:int = 15
    @classmethod
    def from_env(cls) -> "ImportConfig":
        """创建一个ImportConfig对象
        因为上面的很多参数都会去env里读取
        所以这里直接cls（）就够了"""
        return cls()
    def get_minio_base_url(self)->str:
        """把minIO地址拼成完整的网址
        比如localhost：9000会变成
        http：//localhost：9000"""
        protocol = "https://" if self.minio_secure else "http://"
        return protocol + self.minio_endpoint
    #全项目共用一个配置对象
    #一开始还没有创建配置对象，所以用一个none
_config:Optional[ImportConfig]=None
def get_config()->ImportConfig:
    """获取项目的配置对象，第一次调用时候创建一个，后面再调用直接用之前那个不用每次都重新创建"""
    #这里操作的是上面那个全局config
    global _config
    #如果还没创建配置对象，就创建一个
    if _config is None:
        _config = ImportConfig.from_env()
    #返回结果
    return _config
