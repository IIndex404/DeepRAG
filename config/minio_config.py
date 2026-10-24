"""MinIO 对象存储配置。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _read_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    img_dir: str
    secure: bool = False


minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT", ""),
    access_key=os.getenv("MINIO_ACCESS_KEY", ""),
    secret_key=os.getenv("MINIO_SECRET_KEY", ""),
    bucket_name=os.getenv("MINIO_BUCKET_NAME", ""),
    img_dir=os.getenv("MINIO_IMG_DIR", "upload-images"),
    secure=_read_bool("MINIO_SECURE", False),
)
