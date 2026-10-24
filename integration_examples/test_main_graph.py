"""
测试整个知识库查询工作流能不能真正跑通

整个流程：

用户问题
↓
商品名确认
↓
普通向量检索 + HyDE检索 + 联网搜索
↓
RRF融合
↓
Rerank精排
↓
生成最终答案
"""

import sys
import uuid
from pathlib import Path

# 支持直接运行这个测试文件
# 自动把项目根目录加入Python搜索路径
project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from dotenv import load_dotenv

from processor.query_processor.main_graph import KBQueryWorkflow
from processor.query_processor.state import QueryGraphState


# 加载.env里的配置
load_dotenv()


def test_main_graph():
    """
    真正拿一个问题跑完整个知识库查询流程
    """

    # =======================1.准备测试问题=======================

    # 创建一个随机会话ID
    # 可以理解成：这一次聊天的身份证
    session_id = str(uuid.uuid4())

    # 创建一个随机消息ID
    # 可以理解成：这一条用户消息的身份证
    message_id = str(uuid.uuid4())

    # 用户真正问的问题
    original_query = "Brother HAK180烫金机的温度应该设置多少？"


    # =======================2.准备初始状态=======================

    # 给整个LangGraph准备最开始的数据
    initial_state: QueryGraphState = {

        # 这次聊天的ID
        "session_id": session_id,

        # 这条消息的ID
        "message_id": message_id,

        # 用户原始问题
        "original_query": original_query,
    }


    # =======================3.创建整个工作流=======================

    # 创建我们刚刚写好的知识库查询工作流
    workflow = KBQueryWorkflow()


    # =======================4.真正开始执行=======================

    # 把initial_state扔进整个工作流
    # 从第一个节点一直执行到AnswerOutput
    result = workflow.run(
        initial_state=initial_state,
        stream=False
    )


    # =======================5.打印测试结果=======================

    print("\n")
    print("=" * 60)
    print("整个工作流执行完成")
    print("=" * 60)

    # 打印用户原来的问题
    print("\n【用户问题】")
    print(original_query)


    # 打印整个工作流最后的state
    # 方便我们检查每个节点到底往里面放了什么
    print("\n【最终State】")
    print(result)


    # 单独打印最后的大模型回答
    print("\n【最终回答】")
    print(result.get("answer"))


    print("\n")
    print("=" * 60)
    print("测试结束")
    print("=" * 60)


# 直接运行这个文件的时候
# 就执行上面的测试方法
if __name__ == "__main__":
    test_main_graph()