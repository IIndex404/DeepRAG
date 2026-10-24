import sys
from pathlib import Path

# 支持直接执行本文件，不依赖PyCharm额外配置PYTHONPATH。
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph
from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
load_dotenv()
class KBQueryWorkflow:
    def __init__(self):
        self.workflow = StateGraph(QueryGraphState) #1.初始化LangGraph状态图
        self._init_nodes() #2.初始化所有节点
        self._register_nodes() #3.注册节点
        self._setup_routes() #4.设置入口和路由
        self._compiled_app  = None #5.编译后的工作流先设为空
    def _init_nodes(self):
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_answer_output = NodeAnswerOutput()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
    # =======================规定一些节点=======================
    def _register_nodes(self):
        self.workflow.add_node("node_item_name_confirm",self.node_item_name_confirm) #商品具体名字节点加入，add方法是workflow自带的
        self.workflow.add_node("node_multi_search",lambda x:x) #仅仅是创建一个多路搜索节点，收到什么就返回什么
        self.workflow.add_node("node_search_embedding",self.node_search_embedding) #建立一个向量检索节点
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)  #建立一个hype向量检索节点
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp) #建立一个联网搜索节点
        self.workflow.add_node("node_join",lambda x: {}) #普通检索、HyDE检索和联网搜索的结果汇合点
        self.workflow.add_node("node_rrf", self.node_rrf) #RRF融合排序
        self.workflow.add_node("node_rerank",self.node_rerank) #Rerank精排
        self.workflow.add_node("node_answer_output",self.node_answer_output) #最终答案
    # =======================工程化方法=======================
    def _route_after_item_name_confirm(self,state:QueryGraphState) -> str:
        if state.get("answer"):
            return "node_answer_output" #如果前面已经直接生成answer
        return "node_multi_search" #否则继续进入普通检索、HyDE检索和联网搜索
    # =======================搭建整个工作流=======================
    def _setup_routes(self):
        # 1. 入口节点
        self.workflow.set_entry_point("node_item_name_confirm") #1. 流程启动的时候，第一个就跑名字叫node_item_name_confirm的这个节点
        # 2. 商品确认后的条件分支
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm, #这里是条件判断生成节点，单独写一个方法是可读性、可维护性、调试性都更好。
        {   "node_answer_output":"node_answer_output", #有的话直接输出结果answer_output
            "node_multi_search":"node_multi_search" })#没有的话就生成条件边进入多线路搜索节点
        # 3. 同时进行普通检索、HyDE检索和联网搜索
        self.workflow.add_edge("node_multi_search","node_search_embedding") #拿着用户问题进入向量检索节点
        self.workflow.add_edge("node_multi_search","node_search_embedding_hyde") #拿着用户问题进入问题向量检索节点
        self.workflow.add_edge("node_multi_search", "node_web_search_mcp") #拿着用户问题进入网络搜索节点
        # 4. 三路结果汇合
        self.workflow.add_edge("node_search_embedding", "node_join") #向量检索节点结果加入中间节点
        self.workflow.add_edge("node_search_embedding_hyde", "node_join") #问题向量检索节点结果加入中间节点
        self.workflow.add_edge("node_web_search_mcp", "node_join") #网络搜索节点结果加入中间节点
        # 5.RBF -> Rerank -> Answer
        self.workflow.add_edge("node_join", "node_rrf") #看里面的相关性
        self.workflow.add_edge("node_rrf", "node_rerank") #看和问题的相关性
        self.workflow.add_edge("node_rerank", "node_answer_output") #结束后输出
        # 6.结束
        self.workflow.add_edge("node_answer_output", END)
        #=======================到这一步整个项目流程搭好了=======================
    #=======================把流程写成一个可以直接用的软件，以后可以直接拿过来用=======================
    def compile(self):
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app
    def run(self,initial_state:QueryGraphState,stream:bool=False) -> QueryGraphState: #这是最后的执行方法
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)

if __name__ == "__main__":
    from utils.mongo_history_utils import get_history_mongo_tool

    try:
        get_history_mongo_tool()
    except ConnectionError as exc:
        logger.error("启动自检失败：%s", exc)
        raise SystemExit(1) from exc

    KBQueryWorkflow().compile()
    logger.info(
        "启动自检通过：MongoDB连接正常，知识库查询工作流编译成功；"
        "实际问答请通过应用入口传入 initial_state。"
    )
