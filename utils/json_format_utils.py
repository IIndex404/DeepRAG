import json
from typing import Any
from bson import ObjectId
class CustomJSONEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，支持 MongoDB ObjectId 等特殊类型
    当 json.dumps 遇到它不认识的类型时，会自动调用 default 方法。
    """
    def default(self, obj:Any) -> Any:
        if isinstance(obj, ObjectId):
            return str(obj)  #如果这个对象是 MongoDB 的 ObjectId，就转成字符串
        else:
            return super().default(obj) #其他不认识的类型，交回给父类去处理（该报错还是报错）
def serialize_json(data: Any,indent:int = 4,ensure_ascii:bool = False) -> str:
        #把 Python 对象转成好看的 JSON 字符串，中文不转义，默认缩进 4 空格。
        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent,cls=CustomJSONEncoder)
def format_json(data: Any,indent:int = 4,ensure_ascii:bool = False) -> str:
        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)