import base64
import logging
import mimetypes
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from minio import Minio
from minio.deleteobjects import DeleteObject

from config.lm_config import lm_config
from config.minio_config import minio_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import FileProcessingError, StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """Markdown 图片处理节点：多模态图片理解。"""

    name = "node_md_img"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理 Markdown 中的本地图片：
        1. 读取 MD 内容、MD 路径和图片目录；
        2. 找出 MD 实际引用的图片；
        3. 调用视觉模型生成图片摘要；
        4. 上传图片到 MinIO，并替换 MD 中的图片地址；
        5. 保存为新的 MD 文件并更新 state。
        """
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        if not images_dir.exists():
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        target_images = self._step_2_scan_images(md_content, images_dir)
        if not target_images:
            self.logger.info("未检测到 MD 中引用的图片，跳过图片处理")
            return state

        summaries = self._step_3_generate_summaries(
            md_path_obj.stem,
            target_images,
        )

        new_md_content = self._step_4_upload_and_replace(
            md_path_obj.stem,
            target_images,
            summaries,
            md_content,
        )

        new_md_file_name = self._step_5_backup_new_md_file(
            state["md_path"],
            new_md_content,
        )

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state

    def _step_1_get_content(
        self,
        state: ImportGraphState,
    ) -> Tuple[str, Path, Path]:
        """从 state 中取得并检查 MD 内容、MD 路径和图片目录。"""
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(
                field_name="md_path",
                expected_type=str,
            )

        md_content = state.get("md_content")
        if not isinstance(md_content, str):
            raise StateFieldError(
                field_name="md_content",
                expected_type=str,
            )

        md_path_obj = Path(md_path)
        if not md_path_obj.is_file():
            raise FileProcessingError(
                message=f"MD 文件 {md_path_obj} 不存在或不是文件"
            )

        images_dir = md_path_obj.parent / "images"
        return md_content, md_path_obj, images_dir

    def _step_2_scan_images(
        self,
        md_content: str,
        images_dir: Path,
    ) -> List[Tuple[str, str, Tuple[str, str]]]:
        """筛选格式受支持且确实被 Markdown 引用的图片。"""
        target_images = []

        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持，跳过：{image_file}")
                continue

            image_path_obj = images_dir / image_file
            if not image_path_obj.is_file():
                continue

            context = self._find_image_in_md(md_content, image_file)
            if not context:
                self.logger.warning(f"图片未在 MD 中引用，跳过：{image_file}")
                continue

            target_images.append(
                (image_file, str(image_path_obj), context)
            )

        return target_images

    def _find_image_in_md(
        self,
        md_content: str,
        image_file: str,
        context_len: int = 100,
    ) -> Optional[Tuple[str, str]]:
        """查找图片引用，并返回引用位置前后的文本。"""
        pattern = re.compile(
            r"!\[.*?\]\(.*?"
            + re.escape(image_file)
            + r".*?\)",
            flags=re.IGNORECASE,
        )

        match = pattern.search(md_content)
        if not match:
            return None

        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        return pre_text, post_text

    def _step_3_generate_summaries(
        self,
        doc_stem: str,
        target_images: List[Tuple[str, str, Tuple[str, str]]],
    ) -> Dict[str, str]:
        """依次调用视觉模型，为每张图片生成中文摘要。"""
        summaries = {}
        request_times: Deque[float] = deque()

        for image_file, image_path, context in target_images:
            self._apply_api_rate_limit(
                request_times,
                max_requests=10,
            )
            summaries[image_file] = self._summarize_image(
                image_path,
                root_folder=doc_stem,
                image_content=context,
            )

        return summaries

    def _apply_api_rate_limit(
        self,
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60,
    ) -> None:
        """通过滑动时间窗口限制 API 请求频率。"""
        current_time = time.time()

        while (
            request_times
            and current_time - request_times[0] >= window_seconds
        ):
            request_times.popleft()

        if len(request_times) >= max_requests:
            sleep_duration = window_seconds - (
                current_time - request_times[0]
            )
            if sleep_duration > 0:
                self.logger.info(
                    f"触发 API 速率限制：{window_seconds} 秒内最多 "
                    f"{max_requests} 次，需要等待 {sleep_duration:.2f} 秒"
                )
                time.sleep(sleep_duration)
                current_time = time.time()

                while (
                    request_times
                    and current_time - request_times[0] >= window_seconds
                ):
                    request_times.popleft()

        request_times.append(current_time)
        self.logger.info(
            f"API 请求时间戳已记录，当前 {window_seconds} 秒窗口内"
            f"请求数：{len(request_times)}"
        )

    def _summarize_image(
        self,
        image_path: str,
        root_folder: str,
        image_content: Tuple[str, str],
    ) -> str:
        """调用视觉模型总结一张图片的内容。"""
        with open(image_path, "rb") as image_file_obj:
            base64_image = base64.b64encode(
                image_file_obj.read()
            ).decode("utf-8")

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

        try:
            chat_model = ChatOpenAI(
                model=lm_config.vl_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f'这是“{root_folder}”文件中的一张图片，'
                                f'图片上文为“{image_content[0]}”，'
                                f'下文为“{image_content[1]}”。'
                                "请用中文简要总结图片内容，用作 Markdown 图片标题。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{mime_type};base64,{base64_image}"
                                )
                            },
                        },
                    ],
                }
            ]

            response = chat_model.invoke(messages)
            return str(response.content).strip().replace("\n", " ")
        except Exception as error:
            self.logger.error(f"图像总结失败：{image_path}，错误：{error}")
            return "图片描述"

    def _step_4_upload_and_replace(
        self,
        doc_stem: str,
        target_images: List[Tuple[str, str, Tuple[str, str]]],
        summaries: Dict[str, str],
        md_content: str,
    ) -> str:
        """上传图片、合并摘要与 URL，并替换 Markdown 图片引用。"""
        minio_client = get_minio_client()
        upload_dir = (
            f"{minio_config.img_dir}/{doc_stem}"
        ).replace(" ", "")

        self._clean_minio_directory(minio_client, upload_dir)

        urls = self._upload_images_batch(
            minio_client,
            upload_dir,
            target_images,
        )
        image_info = self._merge_summary_and_url(summaries, urls)
        return self._process_md_file(md_content, image_info)

    def _clean_minio_directory(
        self,
        minio_client: Minio,
        prefix: str,
    ) -> None:
        """上传前删除该文档原来保存在 MinIO 中的旧图片。"""
        try:
            objects_to_delete = minio_client.list_objects(
                minio_config.bucket_name,
                prefix=prefix,
                recursive=True,
            )
            delete_list = [
                DeleteObject(item.object_name)
                for item in objects_to_delete
            ]

            if delete_list:
                errors = minio_client.remove_objects(
                    minio_config.bucket_name,
                    delete_list,
                )
                for delete_error in errors:
                    self.logger.error(f"删除 MinIO 文件失败：{delete_error}")
        except Exception as error:
            self.logger.error(f"清理 MinIO 目录失败：{error}")

    def _upload_images_batch(
        self,
        minio_client: Minio,
        upload_dir: str,
        target_images: List[Tuple[str, str, Tuple[str, str]]],
    ) -> Dict[str, Optional[str]]:
        """批量上传图片，返回图片文件名与访问 URL 的对应关系。"""
        urls: Dict[str, Optional[str]] = {}

        for image_file, image_path, _ in target_images:
            object_name = f"{upload_dir}/{image_file}"
            urls[image_file] = self._upload_to_minio(
                minio_client,
                image_path,
                object_name,
            )

        return urls

    def _upload_to_minio(
        self,
        minio_client: Minio,
        local_path: str,
        object_name: str,
    ) -> Optional[str]:
        """上传一张图片，成功时返回可访问 URL，失败时返回 None。"""
        try:
            content_type = (
                mimetypes.guess_type(local_path)[0]
                or "application/octet-stream"
            )
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=local_path,
                content_type=content_type,
            )

            protocol = "https" if minio_config.secure else "http"
            base_url = (
                f"{protocol}://{minio_config.endpoint}/"
                f"{minio_config.bucket_name}"
            )
            return f"{base_url}/{object_name}"
        except Exception as error:
            self.logger.error(
                f"图片上传 MinIO 失败：{local_path}，错误：{error}"
            )
            return None

    def _merge_summary_and_url(
        self,
        summaries: Dict[str, str],
        urls: Dict[str, Optional[str]],
    ) -> Dict[str, Tuple[str, str]]:
        """把同一张图片的摘要和访问 URL 合并到一起。"""
        image_info = {}

        for image_file, summary in summaries.items():
            image_url = urls.get(image_file)
            if image_url:
                image_info[image_file] = (summary, image_url)

        return image_info

    def _process_md_file(
        self,
        md_content: str,
        image_info: Dict[str, Tuple[str, str]],
    ) -> str:
        """用图片摘要和 MinIO URL 替换本地 Markdown 图片引用。"""
        for image_file, (summary, new_url) in image_info.items():
            pattern = re.compile(
                r"!\[.*?\]\(.*?"
                + re.escape(image_file)
                + r".*?\)",
                flags=re.IGNORECASE,
            )
            md_content = pattern.sub(
                lambda _: f"![{summary}]({new_url})",
                md_content,
            )

        self.logger.info(
            f"MD 文件图片引用替换完成，共替换 "
            f"{len(image_info)} 处图片引用"
        )
        return md_content

    def _step_5_backup_new_md_file(
        self,
        origin_md_path: str,
        md_content: str,
    ) -> str:
        """把处理后的内容写入同目录下的“原文件名_new.md”。"""
        new_md_file_name = (
            os.path.splitext(origin_md_path)[0]
            + "_new.md"
        )

        with open(new_md_file_name, "w", encoding="utf-8") as file_obj:
            file_obj.write(md_content)

        self.logger.info(f"处理后的 MD 已保存：{new_md_file_name}")
        return new_md_file_name
