"""
掌柜智库查询Web服务
作用：
用户提问
↓
POST /query
↓
调用KBQueryWorkflow
↓
普通检索 + HyDE + 联网搜索
↓
RRF
↓
Rerank
↓
生成最终答案
↓
SSE流式返回给前端
另外还能：
1.查看历史聊天
2.清空历史聊天
3.以后打开chat.html
"""
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, List
# 删掉上面两行，改成下面这样
from fastapi.responses import FileResponse

# ============================================================
# 1.先把knowledge_base加入Python搜索路径
# ============================================================
# 当前文件：
# knowledge_base/web/api/query_service.py
# parents[0] = api
# parents[1] = web
# parents[2] = knowledge_base
PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0,project_root_str) #给 Python 指路，把项目根目录加到搜索清单最前面
# ============================================================
# 2.导入FastAPI需要的东西
# ============================================================
import uvicorn
from fastapi import (BackgroundTasks,FastAPI,HTTPException,Request,)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse,StreamingResponse
from pydantic import BaseModel,Field
# ============================================================
# 3.导入我们自己项目已经写好的工具
# ============================================================
from processor.query_processor.main_graph import KBQueryWorkflow
from processor.import_processor.base import setup_logging
from tool.logger import logger
from utils.mongo_history_utils import (clear_history,get_recent_messages,)
from utils.sse_utils import (SSEEvent,create_sse_queue,push_to_session,sse_generator,)
# ============================================================
# 4.前端目录
# ============================================================
# 以后：knowledge_base/front/chat.html
FRONT_DIR = PROJECT_ROOT / "web" / "page" #前端的所有 HTML 文件，以后统统都放在 F:\00ZhangGuiZhiKu\front 这个文件夹里，别放错地方了。”
# ============================================================
# 5.规定POST /query传进来的数据格式
# ============================================================
class QueryRequest(BaseModel):
    """
    前端提问时传：
    {
        "query": "联想海豚打印机怎么加墨？",
        "session_id": null,
        "is_stream": true
    }
    第一次聊天：
    session_id可以不传
    后面继续聊天：
    一直使用之前的session_id
    """
    # 用户的问题
    query: str = Field(...,description="用户查询内容")
    # 当前聊天窗口ID，第一次可以不传，后端自动生成
    session_id:Optional[str]=Field(None,description="会话ID，不传则自动生成")
    # 是否使用SSE流式回答
    is_stream:bool=Field(True,description="是否流式返回")
# ============================================================
# 6.普通回答返回格式
# ============================================================
class QueryResponse(BaseModel):
    message: str # 告诉前端：问题已经收到
    session_id: str # 当前聊天窗口ID
    answer: str = ""  # 大模型最终答案
# ============================================================
# 11.流式提问提交成功后返回什么
# ============================================================
class StreamSubmitResponse(BaseModel):
    """
    如果：
    is_stream = True
    POST /query不会等大模型回答完。
    它先马上返回：
    {
        "message": "Query submitted",
        "session_id": "abc123",
        "task_id": "xxx"
    }
    然后真正的回答在后台继续生成。
    """
    message: str # 告诉前端：问题已经收到
    session_id: str # 当前聊天窗口ID
    task_id: str # 这一次提问自己的ID
# ============================================================
# 12.查询业务类
# ============================================================
class QueryService:
    """
    这个类就是专门干“查询相关工作”的。
    主要干4件事：
    1.生成session_id
    2.生成task_id
    3.调用KBQueryWorkflow回答问题
    4.查询/删除MongoDB聊天历史
    """
    # ========================================================
    # 12.1 生成聊天窗口ID
    # ========================================================
    def generate_session_id(self)->str:
        """
        session_id就是“聊天窗口身份证”。
        uuid.uuid4()
        会生成类似：
        7c92e2ce-cc4d-4ef3-91da-123456789abc
        基本不会重复。
        """
        return str(uuid.uuid4())
    # ========================================================
    # 12.2 生成本次提问任务ID
    # ========================================================
    def generate_task_id(self) -> str:
        """
        session_id：
        代表整场聊天
        task_id：
        代表这一次提问
        比如：
        session_id = 聊天窗口A
        task_id1 = 第一问
        task_id2 = 第二问
        task_id3 = 第三问
        """
        return str(uuid.uuid4())
    # ========================================================
    # 12.3 准备查询工作流最开始的state
    # ========================================================
    def build_initial_state(self,session_id:str,user_query:str,is_stream:bool)->Dict:
        """
        我们之前的KBQueryWorkflow所有节点
        都是靠state传数据。
        所以工作流开始之前，
        先准备一个最初的state。
        后面节点会不停修改它。
        """
        return {
            # 当前是哪一个聊天窗口
            "session_id": session_id,
            # 用户消息ID，现在还没存MongoDB，所以先空着
            "message_id": "",
            # 用户刚刚真正输入的问题
            "original_query": user_query,
            # ================================================
            # 下面三个先都是空
            # 因为搜索节点还没有执行
            # ================================================
            # 普通向量搜索找到的切片
            "embedding_chunks": [],
            # HyDE搜索找到的切片
            "hyde_embedding_chunks": [],
            # 联网搜索找到的内容
            "web_search_docs": [],
            # ================================================
            # 排序结果也先是空
            # ================================================
            # 三路搜索经过RRF融合后的结果
            "rrf_chunks": [],
            # 再经过Rerank精排后的结果
            "reranked_docs": [],
            # ================================================
            # 最终生成相关数据
            # ================================================
            # 最终喂给大模型的Prompt，现在还没拼，所以先空
            "prompt": "",
            # 最终答案，现在当然也没有
            "answer": "",
            # ================================================
            # 商品名和问题改写
            # ================================================
            # 用户到底问的是哪个商品NodeItemNameConfirm后面会填
            "item_names": [],
            # 用户问题经过补全后的完整问题，比如：用户：“它怎么加墨？”改写成：“联想海豚打印机怎么加墨？”
            "rewritten_query": "",
            # MongoDB里以前的聊天
            "history": [],
            # True：答案生成一点就推一点，False：最后一次性拿完整答案，
            "is_stream":is_stream
        }
    # ========================================================
    # 12.4 真正运行查询工作流
    # ========================================================
    def run_query_graph(
            self,
            session_id:str,
            user_query:str,
            is_stream:bool)->Dict:
        """
        这个方法非常重要。
        用户问题
        ↓
        组装state
        ↓
        创建KBQueryWorkflow
        ↓
        把state扔进去
        ↓
        整条查询流程开始跑
        ↓
        最后得到final_state
        """
        # --------------------------------------------
        # 第1步：
        # 先准备最开始的state
        # --------------------------------------------
        state=self.build_initial_state(session_id=session_id,user_query=user_query,is_stream=is_stream)
        # --------------------------------------------
        # 第2步：
        # 创建我们前面写好的查询工作流
        # --------------------------------------------
        workflow = KBQueryWorkflow()
        # --------------------------------------------
        # 第3步：
        # 把state交给查询工作流执行
        # --------------------------------------------
        # 这个stream是LangGraph自己的stream，我们现在不需要LangGraph流式返回state，所以写False。
        final_state = workflow.run(initial_state=state,stream=False)
        # 整个LangGraph跑完以后
        # 返回最后的state
        return final_state
    # ========================================================
    # 12.5 查MongoDB聊天记录
    # ========================================================
    def get_history(self,session_id:str,limit:int=50)->List[Dict]:
        """
        根据session_id：
        去MongoDB找这个聊天窗口以前的消息。
        """
        # --------------------------------------------
        # 第1步：
        # 去MongoDB拿消息
        # --------------------------------------------
        records = get_recent_messages(session_id=session_id,limit = limit,)
        # 最后准备返回给前端的数据
        result = []
        # --------------------------------------------
        # 第2步：
        # 把MongoDB消息一条一条整理
        # --------------------------------------------
        for record in records:
            """
            MongoDB原始数据大概：
            {
                "_id": ObjectId(...),
                "session_id": "abc",
                "role": "user",
                "text": "怎么加墨？",
                ...
            }
            但是ObjectId不能直接变成JSON。所以：
            ObjectId
            ↓
            str
            """
            result.append({
                "_id": str(record.get("_id","")), # MongoDB消息ID
                "session_id": record.get("session_id",""),  # 属于哪个聊天窗口
                "role": record.get("role",""), # user还是assistant
                "text": record.get("text",""), # 真正聊天内容
                "rewritten_query":record.get("rewritten_query",""), # 改写后的完整问题
                "item_names":record.get("item_names",[]),  # 这条消息对应哪些商品
                "image_urls":record.get("image_urls",[]), # 回答里面相关的图片
                "ts": record.get("ts") # 这条消息是什么时候产生的
                })
        return result
    # ========================================================
    # 12.6 删除某个聊天窗口的历史
    # ========================================================
    def clear_history(self,session_id:str)->int:
        """
        比如：session_id = abc123
        就把MongoDB里面：session_id = abc123的聊天消息全部删掉。
        """
        return clear_history(session_id)
# ============================================================
# 13.创建一个QueryService对象
# ============================================================
"""
以后整个FastAPI都使用这个对象：
query_service
这样下面不用每次都：
QueryService()
QueryService()
QueryService()
"""
query_service = QueryService()
# ============================================================
# 14.创建FastAPI应用
# ============================================================
"""
可以把app理解成：整个8001查询服务器本体。
"""
app = FastAPI(title="掌柜智库",description="知识库查询服务",version="1.0.0",)
# ============================================================
# 15.允许前端访问这个后端
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 暂时任何来源都允许访问
    allow_credentials=False, # 现在没有做Cookie登录
    allow_methods=["*"], # GET / POST / DELETE等都允许
    allow_headers=["*"], # 请求头也全部允许
)
# ============================================================
# 16.把front文件夹交给FastAPI
# ============================================================
"""
如果：
knowledge_base/front
这个文件夹存在，
就把它挂到：
/front
以后例如：
knowledge_base/front/xxx.js
浏览器就可以通过：
/front/xxx.js访问。
"""
if FRONT_DIR.exists():
    app.mount("/front", StaticFiles(directory=FRONT_DIR), name="front")
# ============================================================
# 17.最简单的测试接口
# ============================================================
@app.get("/")
async def root():
    """
    启动以后浏览器访问：
    http://127.0.0.1:8001/
    如果看到：
    {
        "message": "掌柜智库查询服务运行正常"
    }
    就说明FastAPI本身启动成功。
    """
    return {"message": "掌柜智库查询服务运行正常"}
# ============================================================
# 18.打开chat.html
# ============================================================
@app.get("/chat.html")
async def chat_page():
    """
    以后前端写好：
    knowledge_base/front/chat.html
    浏览器访问：
    http://127.0.0.1:8001/chat.html
    就把这个HTML网页返回给浏览器。
    """
    chat_html = FRONT_DIR/"chat.html"
    if not chat_html.exists():
        raise HTTPException(status_code=404,detail="front/chat.html还没有创建")
    return FileResponse(path=str(chat_html))
# ============================================================
# 19.POST /query
# 真正接收用户问题
# ============================================================
@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    这是整个查询Web服务最重要的接口。
    比如前端发送：POST /query
    {
        "query": "联想海豚打印机怎么加墨？",
        "session_id": null,
        "is_stream": true
    }然后这里开始处理。
    """
    # ========================================================
    # 第1步：
    # 把用户问题拿出来
    # ========================================================
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="query不能为空")
    # ========================================================
    # 第2步：
    # 确定session_id
    # ========================================================
    """
    两种情况：
    情况1：用户第一次聊天。
    前端：
    session_id = None
    那我们就自动创建一个新session_id。
    情况2：
    已经聊过几轮了。
    前端：
    session_id = abc123
    那么继续使用abc123。
    这样历史上下文才能接得上。
    """
    session_id = (request.session_id or query_service.generate_session_id())
    # ========================================================
    # 第3步：
    # 给这一次提问创建一个task_id
    # ========================================================
    """
    session_id：
    整个聊天窗口身份证
    task_id：
    当前这一问身份证
    """
    task_id = query_service.generate_task_id()
    # ========================================================
    # 第4步：
    # 如果用户要求流式回答
    # ========================================================
    if request.is_stream:
        """
        流式回答不能：
        POST /query
        ↓
        一直卡在那里
        ↓
        等几十秒
        ↓
        最后返回
        而是：
        POST /query
        ↓
        马上告诉前端：
        “收到问题了”
        ↓
        后台慢慢生成答案
        ↓
        SSE一点一点把答案推给前端
        """
        # --------------------------------------------
        # 先给当前聊天窗口创建一个SSE消息队列
        # --------------------------------------------
        # 给当前聊天会话创建一个专属消息队列。
        create_sse_queue(session_id)
        # --------------------------------------------
        # 把真正耗时的查询丢到后台执行
        # --------------------------------------------
        """
        BackgroundTasks大白话：
        FastAPI先给用户返回结果。
        然后：
        后台继续干这个慢活。
        所以用户不用卡在POST /query这里等。
        """
        background_tasks.add_task(
            _run_stream_query, # 后台要执行这个方法
            session_id,# 把session_id传进去
            user_query,# 把用户问题传进去
        )
        # --------------------------------------------
        # 先马上告诉前端：
        # 问题已经收到
        # --------------------------------------------
        return StreamSubmitResponse(
            message="Query submitted",
            session_id=session_id, # 前端后面还要继续使用这个session_id
            task_id=task_id,  # 当前问题对应的任务ID
        )
    # ========================================================
    # 第5步：
    # 如果不是流式回答
    # ========================================================
    """
    is_stream = False
    就简单很多：
    用户问题
    ↓
    整个LangGraph跑完
    ↓
    拿最终answer
    ↓
    直接return
    """
    try:
        # 真正执行整个查询工作流
        final_state = query_service.run_query_graph(
            session_id = session_id,
            user_query = user_query,
            is_stream = False,)
        answer = final_state.get("answer") or ""
        # 一次性返回最终答案
        return QueryResponse(
            message = "处理完成" , # 告诉前端：问题已经收到
            session_id = session_id ,# 当前聊天窗口ID
            answer = answer ) # 大模型最终答案
    except Exception as e:
        logger.error("查询失败：%s",e)
        raise HTTPException(status_code=500,datail=f"查询失败：{e}")
# ============================================================
# 20.后台真正执行流式查询
# ============================================================
def _run_stream_query(session_id, user_query)->None:
    """
    前面的POST /query：
    只负责：
    “收到问题，我先返回。”
    真正耗时间的：
    商品名确认
    向量检索
    HyDE
    联网搜索
    RRF
    Rerank
    大模型生成答案
    都在这里后台执行。
    """
    try:
        # --------------------------------------------
        # 第1步：
        # 先往SSE队列塞一条“正在处理”
        # --------------------------------------------
        push_to_session(
            session_id = session_id, # 塞进哪个聊天窗口的队列
            event = SSEEvent.PROGRESS,  # 这条消息属于“进度消息”
            data =  {"status": "processing","message": "正在处理问题",} ) # 真正消息内容
        # --------------------------------------------
        # 第2步：
        # 真正跑查询工作流
        # --------------------------------------------
        query_service.run_query_graph(
            session_id = session_id,
            user_query = user_query,
            is_stream = True,)
        """
        到NodeAnswerOutput的时候：
        大模型每生成一点文字
        ↓
        push_to_session()
        ↓
        放到当前session_id的队列
        ↓
        SSE接口拿出去
        ↓
        前端显示
        """
    except Exception as e:
        logger.exception("查询流程执行失败：%s",e)
        # 不只是后台打印报错
        # 还要把报错通过SSE告诉前端
        push_to_session(session_id, SSEEvent.ERROR,
        {
                "status": "error",
                "message": str(e),
            })
# ============================================================
# 21.GET /stream/{session_id}
# 前端专门通过这个接口接收流式文字
# ============================================================
@app.get("/stream/{session_id}")
async def stream(session_id:str, request: Request):
    """
       这个接口不要和POST /query混淆。
       POST /query：
       负责“提交问题”。
       GET /stream/{session_id}：
       负责“接收一点一点生成的答案”。
       举例：
       GET /stream/abc123
       后端就一直盯着：
       abc123这个SSE队列。
       队列出现：
       delta = "联想"
       ↓
       发给前端
       队列又出现：
       delta = "海豚打印机"
       ↓
       再发给前端
       直到收到：
       FINAL
       表示回答彻底结束。
       """
    return StreamingResponse(
        sse_generator(session_id ,request), # 这个生成器会一直从当前session队列拿消息

    # 告诉浏览器：
    # 这是SSE，不是普通JSON
    media_type = "text/event-stream", # 告诉浏览器：这是SSE，不是普通JSON
    headers = {"Cache-Control": "no-cache",
               "Connection": "keep-alive",})
# ============================================================
# 22.GET /history/{session_id}
# 查询聊天历史
# ============================================================
@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50):
    try:
        items = query_service.get_history(session_id,limit=limit)
        # 返回给前端
        return{
            "session_id": session_id,
            "items": items,
        }
    except Exception as e:
        logger.exception("获取历史记录失败：%s", e)
        raise HTTPException(
            status_code=500,
            detail=f"history error: {e}",
        )
# ============================================================
# 23.DELETE /history/{session_id}
# 删除聊天历史
# ============================================================
@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """
    比如：
    DELETE /history/abc123
    就相当于：
    MongoDB里面找到所有：
    session_id = abc123
    然后全部删除。
    """
    count = query_service.clear_history(session_id)
    # 告诉前端删除结果
    return {
        "message": "History cleared",
        "deleted_count": count,
    }
# ============================================================
# 24.直接运行这个文件时，启动8001服务
# ============================================================
if __name__ == "__main__":
    """
    当你直接运行：
    query_service.py
    才会执行这里。
    然后启动：
    http://127.0.0.1:8001
    """
    # 开启项目日志
    setup_logging()
    # 启动FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8001)
