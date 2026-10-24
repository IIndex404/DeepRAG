"""MinIO 客户端初始化工具。"""

from typing import Optional

from minio import Minio

from config.minio_config import minio_config


_minio_client: Optional[Minio] = None


def _validate_minio_config() -> None:
    required_values = {
        "MINIO_ENDPOINT": minio_config.endpoint,
        "MINIO_ACCESS_KEY": minio_config.access_key,
        "MINIO_SECRET_KEY": minio_config.secret_key,
        "MINIO_BUCKET_NAME": minio_config.bucket_name,
    }
    missing_names = [
        name for name, value in required_values.items() if not value
    ]
    if missing_names:
        raise ValueError(
            "缺少 MinIO 配置：" + ", ".join(missing_names)
        )


def get_minio_client() -> Minio:
    """取得项目共用的 MinIO 客户端，第一次调用时完成初始化。"""
    global _minio_client

    if _minio_client is not None:
        return _minio_client

    _validate_minio_config()

    client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.secure,
    )

    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)

    # 存储桶访问策略由 MinIO 管理端统一配置。
    # 这里不自动把整个 bucket 改成匿名公开，避免意外暴露其他文件。
    _minio_client = client
    return _minio_client
