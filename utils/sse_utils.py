"""
SSE流式输出工具
作用：
1. 给每个session_id创建一个自己的消息队列
2. 后端业务节点可以往这个队列里塞消息
3. SSE接口不断从队列里拿消息
4. 再把消息实时推给前端

业务节点
↓
push_to_session()
↓
消息队列
↓
sse_generator()
↓
前端
"""
from queue import Queue
from queue import Empty
from typing import Dict,Any,List
import asyncio
import json
# ============================================================
# SSE事件类型
# ============================================================
class SSEEvent:
    #告诉前端：当前推过去的这条消息到底是什么类型
    READY = "ready" # SSE连接刚刚建立成功
    PROGRESS = "progress" # 后端处理进度发生变化
    DELTA = "delta"  # 大模型新生成了一小段文字
    FINAL = "final"  # 整个回答彻底生成结束
    ERROR = "error" # 后端处理出错
    # ============================================================
    # 保存所有会话对应的消息队列
    # ============================================================
_sse_queues:Dict[str,Queue]={} # 每个聊天session都有自己的队列,这样A用户的消息不会跑到B用户那里
# ============================================================
# 1.给某个session创建消息队列
# ============================================================
def create_sse_queue(session_id:str) -> Queue: #给当前聊天会话创建一个专属消息队列。
    # 比如：session_id = "abc123"最后：_sse_queues = {"abc123": Queue()}
    queue=Queue() # 创建一个新的消息队列
    _sse_queues[session_id]=queue # 用session_id作为key保存起来
    return queue
# ============================================================
# 2.拿到某个session的消息队列
# ============================================================
def get_sse_queue(session_id:str)-> Queue|None:
    return _sse_queues.get(session_id)
# ============================================================
# 3.往指定session的队列里面塞消息
# ============================================================
def push_to_session(session_id:str,event:str,data:Dict[str,Any])->bool:
    queue = _sse_queues.get(session_id)  # 先根据session_id找到这个会话的队列
    if queue is None:
        return False
    message = {"event":event,"data":data }
    queue.put(message) # 把消息塞进当前session的队列
    return True
# ============================================================
# 4.把消息包装成SSE规定的字符串格式
# ============================================================
def _format_sse_message(event:str,data:Dict[str,Any]) -> str:
    """
    SSE要求后端发给前端的数据长这样：
    event: delta
    data: {"delta": "HAK180"}
    最后还必须有两个换行。
    """
    json_data = json.dumps(data, ensure_ascii=False) # 把Python字典转成JSON字符串
    return (f"event:{event}\n"
            f"data:{json_data}\n\n") # 按照SSE规定的格式拼起来
# ============================================================
# 5.SSE真正不断往前端推消息的生成器
# ============================================================
async def sse_generator(session_id:str,request):
    """
    SSE接口真正使用的生成器。
    它会一直：
    从队列拿消息
    ↓
    转成SSE格式
    ↓
    yield给前端
    ↓
    再继续等下一条
    直到：
    FINAL
    ERROR
    或者前端主动断开连接
    """
    # ========================================================
    # 1.找到当前session对应的消息队列
    # ========================================================
    queue = _sse_queues.get(session_id)
    if queue is None:
        queue = create_sse_queue(session_id)     #如果没有队列,顺手创建一个
    # ========================================================
    # 2.先告诉前端：
    # SSE连接已经建立成功
    # ========================================================
    yield _format_sse_message(SSEEvent.READY,{"session_id":session_id})
    try:
        # ====================================================
        # 3.不断等待后端业务节点往队列里塞消息
        # ====================================================
        while True:
            if await request.is_disconnected():
                break
            try:
                message = await asyncio.to_thread(queue.get,True,1)
                # queue.get() 是阻塞的：没消息就死等，会卡死整个程序
                # to_thread：扔到别的线程去等，主线程别被堵住
                # get(True, 1)：没消息就等，最多等 1 秒，超时抛 Empty
            except Empty: #1秒还没有新消息,不算报错,回到while继续等
                continue
            # =================================================
            # 4.拿出消息类型和数据
            # =================================================
            """
            # ===== 队列里塞进去的一条消息 =====
            message = {
                "event": "delta",              # 事件类型
                "data": {"delta": "你好"}      # 具体数据
            }
            """
            event = message.get("event","message") #有event返回它的值，比如 "delta",没有返回默认值 "message"
            data = message.get("data",{})
            # =================================================
            # 5.把消息按照SSE格式推给前端
            # =================================================
            yield _format_sse_message(event,data)
            # =================================================
            # 6.如果已经FINAL或者ERROR
            # 说明这次流式输出结束
            # =================================================
            if event in(SSEEvent.FINAL,SSEEvent.ERROR):
                break
    finally:
        # ====================================================
        # 7.连接结束以后清理这个session的队列
        # 防止一直占内存
        # ====================================================
        _sse_queues.pop(session_id,None)
""""
============================================================
流程：
_sse_queues        保存所有 session 的队列
create_sse_queue   给 session 建队列
push_to_session    业务节点往队列塞消息
sse_generator      循环从队列取消息，yield 给前端
_format_sse_message  把消息转成 SSE 格式
链路：业务节点 → push_to_session → 队列 → sse_generator → 前端 
============================================================
"""