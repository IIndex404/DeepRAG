from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()
@dataclass
class McpConfig: #专门保存MCP相关配置
    mcp_base_url: str #百炼MCP服务地址
    api_key: str  #调用百炼MCP需要的API Key
mcp_config = McpConfig(
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),
    api_key=os.getenv("MCP_DASHSCOPE_API_KEY")
)
