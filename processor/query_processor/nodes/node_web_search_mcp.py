"""
网上搜索节点
"""
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from config.mcp_config import mcp_config
import json
import asyncio
import httpx
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
class NodeWebSearchMcp(NodeBase):
    name:str = "node_web_search_mcp"
    def process(self,state:QueryGraphState)->QueryGraphState:
        """
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        query = state.get("rewritten_query","") #从state里面拿前面已经重写完整的用户问题，没有就给空字符串
        docs = [] #先创建一个空列表，后面专门装整理好的联网搜索结果
        if query: # 如果用户问题不是空的，才真正去联网搜索
             result = asyncio.run(self._mcp_call(query)) #调用下面自己写的的异步_mcp_call，让百炼MCP真正帮我们联网搜索
             if result: #如果MCP真的返回了结果
                 pages = json.loads(result.content[0].text).get("pages",[])
                 #拿MCP返回的正文，把JSON字符串转成字典，再取里面的pages搜索结果列表
                 for item in pages:
                     title = (item.get("title") or "").strip()  # 拿网页标题，没有就给空字符串
                     url = (item.get("url") or "").strip()  # 拿网页链接，没有就给空字符串
                     snippet = (item.get("snippet") or "").strip()  # 拿网页摘要，没有就给空字符串
                     if not snippet: #如果这条网页连摘要都没有,直接跳过这条，不要它
                         continue
                     docs.append(
                         {"title":title,
                          "url":url,
                          "snippet":snippet}) #把这一条网页整理成统一格式以后放进docs列表
                 logger.info(f"MCP 搜索结果: {docs}") #把最终整理好的联网搜索结果打印出来
        if docs: #如果最终确实搜到了有效网页
            return {"web_search_docs":docs} #把这些联网结果放到web_search_docs，交给后面的RRF融合节点
        return{}

    async def _mcp_call(self, query):
        # 先创建HTTP客户端，并把百炼API Key放到请求头里面
        async with httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {mcp_config.api_key}"
                },timeout=10) as http_client:
            # 连接百炼MCP服务器
            async with streamable_http_client(
                    mcp_config.mcp_base_url,
                    http_client=http_client
            ) as (read, write):
                # 建立MCP会话
                async with ClientSession(
                        read,
                        write
                ) as session:# 先和MCP服务器完成初始化握手
                    await session.initialize() # 真正调用百炼的联网搜索工具
                    result = await session.call_tool(
                        "bailian_web_search",
                        {
                            "query": query,
                            "count": 5
                        }
                    ) # 把MCP原始搜索结果返回给process继续处理
                    return result
if __name__ == "__main__": #只有直接运行当前py文件的时候才执行下面测试

    init_state = {
        "rewritten_query":
            "关于brother HAK180烫金机，如何调节转印温度？"
    } #模拟前面节点已经帮我们重写好的用户问题

    node_web_search_mcp = NodeWebSearchMcp() #实例化一个联网搜索节点

    result = node_web_search_mcp(
        init_state
    ) #把测试state交给联网搜索节点跑一遍

    logger.info(
        serialize_json(
            result,
            indent=4
        )
    ) #把最终web_search_docs整理成漂亮JSON打印出来