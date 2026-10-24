"""把 PDF 交给 MinerU 解析为 Markdown。"""

from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import (
    ConfigurationError,
    FileProcessingError,
    PdfConversionError,
    StateFieldError,
)
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """PDF 转 Markdown 节点，配置统一从 config/mineru_config.py 读取。"""

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        md_path = self._step_3_download_and_extract(
            zip_url=zip_url,
            output_dir_obj=output_dir_obj,
            pdf_stem=pdf_path_obj.stem,
        )

        state["md_path"] = md_path
        state["md_content"] = Path(md_path).read_text(encoding="utf-8")
        return state

    def _step_1_validate_paths(
        self,
        state: ImportGraphState,
    ) -> tuple[Path, Path]:
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", expected_type=str)

        pdf_path_obj = Path(pdf_path).expanduser().resolve()
        output_dir_obj = Path(file_dir).expanduser().resolve()
        if not pdf_path_obj.is_file():
            raise FileProcessingError(message=f"PDF 文件不存在：{pdf_path_obj}")
        if pdf_path_obj.suffix.lower() != ".pdf":
            raise FileProcessingError(message=f"文件不是 PDF：{pdf_path_obj}")

        output_dir_obj.mkdir(parents=True, exist_ok=True)
        return pdf_path_obj, output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path) -> str:
        token = mineru_config.api_token
        base_url = (mineru_config.base_url or "").rstrip("/")
        if not token:
            raise ConfigurationError(message="缺少 MINERU_API_TOKEN")
        if not base_url:
            raise ConfigurationError(message="缺少 MINERU_BASE_URL")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        response = requests.post(
            f"{base_url}/file-urls/batch",
            headers=headers,
            json={
                "files": [{"name": pdf_path_obj.name}],
                "model_version": "vlm",
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise PdfConversionError(
                message=(
                    "获取 MinerU 上传链接失败："
                    f"HTTP {response.status_code}，{response.text}"
                )
            )

        result = response.json()
        if result.get("code") != 0:
            raise PdfConversionError(message=f"MinerU 返回业务错误：{result}")

        signed_url = result["data"]["file_urls"][0]
        batch_id = result["data"]["batch_id"]
        with pdf_path_obj.open("rb") as file_obj:
            upload_response = requests.put(
                signed_url,
                data=file_obj,
                timeout=120,
            )
        if upload_response.status_code != 200:
            raise PdfConversionError(
                message=(
                    "PDF 上传 MinerU 失败："
                    f"HTTP {upload_response.status_code}，{upload_response.text}"
                )
            )

        self.logger.info("PDF 上传成功，batch_id=%s", batch_id)
        poll_url = f"{base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        timeout_seconds = 600
        poll_interval = 3

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise PdfConversionError(
                    message=f"MinerU 解析超过 {timeout_seconds} 秒，batch_id={batch_id}"
                )
            try:
                poll_response = requests.get(
                    poll_url,
                    headers=headers,
                    timeout=15,
                )
            except requests.RequestException as exc:
                self.logger.warning("MinerU 轮询网络异常，3 秒后重试：%s", exc)
                time.sleep(poll_interval)
                continue

            if poll_response.status_code != 200:
                raise PdfConversionError(
                    message=(
                        "查询 MinerU 任务失败："
                        f"HTTP {poll_response.status_code}，{poll_response.text}"
                    )
                )

            poll_data = poll_response.json()
            if poll_data.get("code") != 0:
                raise PdfConversionError(message=f"MinerU 轮询业务错误：{poll_data}")

            extract_results = poll_data.get("data", {}).get("extract_result") or []
            if not extract_results:
                raise PdfConversionError(message=f"MinerU 未返回解析结果：{poll_data}")

            result_item = extract_results[0]
            data_state = result_item.get("state")
            if data_state == "done":
                zip_url = result_item.get("full_zip_url")
                if not zip_url:
                    raise PdfConversionError(message="MinerU 已完成但未返回 full_zip_url")
                self.logger.info("MinerU 解析完成，耗时 %.1f 秒", elapsed_time)
                return zip_url
            if data_state == "failed":
                raise PdfConversionError(
                    message=f"MinerU 解析失败：{result_item.get('err_msg', '未知错误')}"
                )

            self.logger.info("MinerU 正在处理，状态=%s，已耗时 %.1f 秒", data_state, elapsed_time)
            time.sleep(poll_interval)

    def _step_3_download_and_extract(
        self,
        zip_url: str,
        output_dir_obj: Path,
        pdf_stem: str,
    ) -> str:
        download_response = requests.get(zip_url, timeout=120)
        if download_response.status_code != 200:
            raise PdfConversionError(
                message=f"下载 MinerU ZIP 失败：HTTP {download_response.status_code}"
            )

        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        zip_save_path.write_bytes(download_response.content)

        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            extract_root = extract_target_dir.resolve()
            for member in zip_file_obj.infolist():
                member_path = (extract_root / member.filename).resolve()
                if not member_path.is_relative_to(extract_root):
                    raise PdfConversionError(message=f"ZIP 包含非法路径：{member.filename}")
            zip_file_obj.extractall(extract_target_dir)

        md_candidates = list(extract_target_dir.rglob("full.md"))
        if not md_candidates:
            raise FileProcessingError(
                message=f"MinerU ZIP 中没有找到 full.md：{extract_target_dir}"
            )

        target_md_file = md_candidates[0]
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        if new_md_path.exists() and new_md_path != target_md_file:
            new_md_path.unlink()
        target_md_file.rename(new_md_path)
        self.logger.info("Markdown 已生成：%s", new_md_path)
        return str(new_md_path.resolve())
