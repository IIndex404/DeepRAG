from typing import Dict, Any
_task_results: Dict[str, Dict[str, Any]] = {}   # task_id -> 结果字典
def set_task_result(task_id:str,key:str,value:Any)->None:
    """
    存任务结果。
    比如：set_task_result("abc", "answer", "你好")
    最终：_task_results = {"abc": {"answer": "你好"}}
    """
    if task_id not in _task_results:
        _task_results[task_id] = {}
    _task_results[task_id][key] = value