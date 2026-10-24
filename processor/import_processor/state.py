"""
state只是数据罢了
这里定义state具体是什么规范
"""
from typing import TypedDict,List
import copy
class ImportGraphState(TypedDict):
    """定义state里有哪些字段"""
    task_id:str
    is_md_read_enabled:bool
    is_pdf_read_enabled:bool
    import_file_path:str
    file_dir:str
    pdf_path:str
    md_path:str
    file_title:str
    item_name:str
    md_content:str
    chunks:List
GRAPH_DEFAULT_STATE:ImportGraphState={
    'task_id':"",
    'is_md_read_enabled':False,
    'is_pdf_read_enabled':False,
    'import_file_path':"",
    'file_dir':"",
    'pdf_path':"",
    'md_path':"",
    'file_title':"",
    'item_name':"",
    'md_content':"",
    'chunks':[]
}
def get_default_state()->ImportGraphState:
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
