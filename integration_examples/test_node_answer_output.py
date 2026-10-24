"""
测试 NodeAnswerOutput
看看能不能正常拼Prompt并生成最终答案
"""

from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput


def test_node_answer_output():

    # 1.造一个假的工作流state
    state = {
        "session_id": "test_answer_001",

        # 用户原始问题
        "original_query": "HAK180烫金机温度应该怎么设置？",

        # 前面节点改写后的问题
        "rewritten_query": "HAK180烫金机的推荐工作温度是多少？",

        # 前面识别出来的商品名
        "item_names": [
            "HAK180烫金机"
        ],

        # 模拟Rerank最后留下的参考资料
        "reranked_docs": [
            {
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "温度设置",
                "content": "HAK180烫金机建议根据材料调整温度，常用温度范围为160℃到190℃。",
                "score": 0.95
            },
            {
                "source": "local",
                "chunk_id": "chunk_002",
                "title": "使用注意事项",
                "content": "不同烫金纸和材料需要使用不同温度，建议先低温测试再逐渐升温。",
                "score": 0.88
            }
        ],

        # 模拟历史聊天
        "history": [
            {
                "role": "user",
                "text": "我有一台HAK180烫金机"
            },
            {
                "role": "assistant",
                "text": "好的，请问你想了解什么？"
            }
        ],

        # 先测试非流式，简单一点
        "is_stream": False,

        # 一开始没有答案
        "answer": ""
    }

    # 2.创建答案输出节点
    node = NodeAnswerOutput()

    # 3.执行节点
    result = node.process(state)

    # 4.看看最终Prompt
    print("\n========== 最终Prompt ==========")
    print(result.get("prompt"))

    # 5.看看最终答案
    print("\n========== 最终答案 ==========")
    print(result.get("answer"))

    # 6.简单检查
    assert result.get("answer")

    print("\n========== 测试成功 ==========")


if __name__ == "__main__":
    test_node_answer_output()