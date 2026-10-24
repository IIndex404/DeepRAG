import asyncio
import os

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


load_dotenv()


async def main():

    api_key = os.getenv("MCP_DASHSCOPE_API_KEY")

    mcp_url = (
        "https://dashscope.aliyuncs.com/"
        "api/v1/mcps/WebSearch/mcp"
    )

    print("API_KEY存在:", bool(api_key))
    print("开始连接百炼MCP...")

    # 先创建一个HTTP客户端
    # 把百炼要求的Authorization放到请求头里面
    async with httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {api_key}"
        }
    ) as http_client:

        # 用新版MCP的streamable_http_client连接百炼
        async with streamable_http_client(
            mcp_url,
            http_client=http_client
        ) as (read, write):

            print("HTTP连接成功")

            # 创建MCP会话
            async with ClientSession(
                read,
                write
            ) as session:

                print("开始初始化MCP...")

                # 和百炼MCP完成协议握手
                await session.initialize()

                print("MCP初始化成功")

                # 看百炼到底给我们提供了哪些工具
                tools_result = await session.list_tools()

                print("可用工具：")

                for tool in tools_result.tools:
                    print(
                        tool.name,
                        "-",
                        tool.description
                    )


if __name__ == "__main__":
    asyncio.run(main())