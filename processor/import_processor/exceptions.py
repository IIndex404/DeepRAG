"""
这个文件专门放“导入流程可能出现的各种错误”。
为什么要单独写这个文件？
因为以后项目很大，可能会出现很多不同的问题，比如：
1. .env 配置没写对
2. PDF 解析失败
3. 图片处理失败
4. 文档切片失败
5. 大模型调用失败
6. 向量生成失败
7. Milvus 保存失败
8. MinIO 保存失败
如果这些错误全部都只写 Exception，
以后看到报错的时候，很难马上知道是哪一类问题。
所以这里给不同类型的错误分别起名字。
"""
class ImportProcessError(Exception):
    #初始化这个类，里面有self，message，node_name，cause
    def __init__(
    self,
    message:str,
    node_name:str="",
    cause:Exception=None):
        """
        创建一个导入流程错误。
        message：
        真正想告诉我们的错误信息。
        比如：
        "PDF 文件不存在"
        node_name：
        是哪个节点出的错。
        比如：
        "pdf_parser"
        cause：
        导致这个错误的原始异常。
        比如 Python 原本报的是：
        FileNotFoundError
        那我们可以把原始错误也保存下来，
        以后方便排查。
        """
        #记住哪个节点出错
        self.node_name = node_name
        #记住最开始到底是什么错误
        self.cause = cause
        #调用父类把message交给他
        super().__init__(message)
    def __str__(self):
        #打印出错误来
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(
            super().__str__()
        )
        if self.cause:
            parts.append(f"(原因：{self.cause})")
        return " ".join(parts)
class StateFieldError(ImportProcessError):
    """
    state 里面的数据有问题时使用。

    我们后面 LangGraph 会一直传一个 state。

    比如某个节点规定必须有：

    state["pdf_path"]

    但是实际没有这个字段，
    或者这个字段是空的，
    或者它的类型不对，

    就可以抛出 StateFieldError。
    """
    def __init__(self,
                 node_name:str="",
                 field_name:str="",
                 expected_type:type=None,
                 message:str="",
                 cause:Exception=None):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段{field_name}缺失或无效"
            if expected_type:
                message += f"，期望类型：{expected_type.__name__}"
        #交给父类统一保存错误信息
        super().__init__(
            message,
            node_name=node_name,
            cause=cause
        )
class ConfigurationError(ImportProcessError):
    """配置错误，比如环境变量缺失"""
    pass


class FileProcessingError(ImportProcessError):
    """文件读取、格式等处理错误"""
    pass


class PdfConversionError(FileProcessingError):
    """PDF 解析或转换错误"""
    pass


class ImageProcessingError(FileProcessingError):
    """图片处理错误"""
    pass


class DocumentSplitError(ImportProcessError):
    """文档切片错误"""
    pass


class EmbeddingError(ImportProcessError):
    """文本转向量时出错"""
    pass


class LLMError(ImportProcessError):
    """调用大模型时出错"""
    pass


class StorageError(ImportProcessError):
    """数据存储时出错"""
    pass


class MilvusError(StorageError):
    """Milvus 相关错误"""
    pass


class MinioError(StorageError):
    """MinIO 相关错误"""
    pass


class ValidationError(ImportProcessError):
    """输入数据不符合要求"""
    pass