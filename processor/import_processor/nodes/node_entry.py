"""
入口节点
拿到上传路径然后
检查路径是否存在
判断是pdf还是md
把结果写回state
"""
import json
import logging
from pathlib import Path
from processor.import_processor.base import BaseNode,setup_logging
from processor.import_processor.state import ImportGraphState
from processor.import_processor.exceptions import (
    ValidationError,
    StateFieldError,
    FileProcessingError, ImportProcessError
)
class NodeEntry(BaseNode):
    #入口节点大类，继承之前写好的BaseNode
    name = 'node_entry'
    def process(self,state:ImportGraphState):
        #定义入口方法，并且这个方法参数继承的之前写好的状态大类
        #拿到文件地址
        import_file_path = state.get('import_file_path')
        #没有文件路径就会报错
        if not import_file_path:
            raise StateFieldError(
                field_name="import_file_path",
                expected_type=str
            )
        #文件地址转Path对象
        import_file_path_obj = Path(import_file_path).expanduser().resolve()
        #如果文件路径有但是里面没有文件也会报错
        if not import_file_path_obj.exists():
            raise FileProcessingError(
                message=f"文件{import_file_path_obj.name}不存在"
            )
        #拿到对象转Path就可以判断是什么格式了
        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = False
        state["file_dir"] = str(import_file_path_obj.parent)
        state["import_file_path"] = str(import_file_path_obj)

        suffix = import_file_path_obj.suffix.lower()
        if suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state["pdf_path"] = str(import_file_path_obj)
        elif suffix == ".md":
            state['is_md_read_enabled'] = True
            state["md_path"] = str(import_file_path_obj)
            state["md_content"] = import_file_path_obj.read_text(encoding="utf-8")
        else:
            raise ValidationError(
                message=(f"该文件的后缀格式{import_file_path_obj.suffix}不支持")
            )
        #拿到文件名字，但是不要后缀
        state["file_title"] = import_file_path_obj.stem
        return state
