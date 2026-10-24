"""这个文件专门用来贵姓所有节点都必须怎么工作
以后不管是PDF解析节点
图片理解节点
文档切片节点
商品名识别节点
他们都继承BaseNode
这样大家就不用每个文件都重新写
1.怎么拿配置
2.怎么记录日志
3.怎么开始执行
4.出错了怎么办
"""

from abc import ABC, abstractmethod
from typing import Optional,TypeVar
import logging
import colorlog
from processor.import_processor.config import ImportConfig,get_config
from processor.import_processor.exceptions import ImportProcessError
#T是类型占位符
T = TypeVar('T')
class BaseNode(ABC):
    name:str = "base_node"
    def __init__(
            self,
            config:Optional[ImportConfig]=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")
    def __call__(self,state:T)->T:
        try:
            #节点开始执行
            self.logger.info(
                f"---{self.name} 开始---"
            )
            # 真正执行具体节点的业务逻辑
            result = self.process(state)
            # 节点正常执行完成
            self.logger.info(
                f"---{self.name} 完成---"
            )
            return result
        except Exception as e:
            self.logger.error(
                f"{self.name}执行失败:{e}"
            )
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )
    @abstractmethod
    def process(self,state:T)->T:
        pass
    def log_step(
            self,step_name:str,
            message:str=" "
    ):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)
def setup_logging(
        level:int=logging.INFO
):
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(

            # 日志显示格式
            "%(log_color)s"
            "%(asctime)s - "
            "%(name)s - "
            "%(levelname)s - "
            "%(message)s",
            # 时间格式
            datefmt="%Y-%m-%d %H:%M:%S",
            # 不同等级显示不同颜色
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        )
    )
    logger.handlers.clear()
    logger.addHandler(handler)