"""
知识库导入Web服务
这个文件干的事情很简单：
前端选择PDF或者MD
↓
POST /upload把文件传过来
↓
先把文件保存到本地temp_data
↓
生成一个task_id
↓
后台调用KBImportWorkflow跑完整导入流程
↓
前端不断请求/status/{task_id}
↓
查看导入到哪一步了
"""
import shutil
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict
import uvicorn
# 统一从 FastAPI 导入所有 Web 服务相关的组件
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
# 你自己的项目内部模块
from processor.import_processor.base import setup_logging
from processor.import_processor.main_graph import KBImportWorkflow
from processor.import_processor.state import get_default_state
# ============================================================
# 1.项目路径
# ============================================================
# 当前文件：
# knowledge_base/web/api/import_service.py
#
# parents[0] = api
# parents[1] = web
# parents[2] = knowledge_base
#拿到当前这个Python文件的绝对路径，往上翻三层，找到项目的根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#上传文件临时保存的位置拼接生成临时上传目录的绝对路径
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp_data" / "upload"
# 前端页面的位置
FRONT_DIR = PROJECT_ROOT / "web" / "page" # 前端页面的位置
# 如果上传目录不存在，就自动创建
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# 2.规定允许上传什么文件
# ============================================================
# 当前导入工作流只支持PDF和Markdown
SUPPORTED_SUFFIXES = {".pdf", ".md"}
# ============================================================
# 3.保存所有导入任务的状态
# ============================================================
"""
举个例子：
_task_status = {
    "abc123": {
        "task_id": "abc123",
        "status": "processing",
        "progress": 60,
        "message": "正在文档切片",
        "result": None,
        "error": None
    }
}
以后前端拿task_id来查，
我们就能告诉它任务进行到哪一步了。
"""
_task_status :Dict[str,Dict[str,Any]] = {}
# 因为后台任务和前端查询可能同时操作这个字典，
# 所以加一个锁，防止同时修改时出问题
_task_lock = Lock()
# ============================================================
# 4.更新任务状态
# ============================================================
def _update_task(task_id:str,**kwargs)-> None:
    """
    更新某个任务的信息。
    比如：
    _update_task(
        task_id,
        status="processing",
        progress=50
    )
    """
    # 同一时间只让一个线程修改任务字典
    with _task_lock:
        # 如果这个task_id还不存在
        # 就先创建一个基础任务
        if task_id not in _task_status:
            _task_status[task_id] = {
                "task_id": task_id,
                "status": "processing",
                "progress": 0,
                "message": "等待处理",
                "result": None,
                "error": None
            }
        # 把传进来的新内容更新进去
        _task_status[task_id].update(kwargs)
# ============================================================
# 5.读取任务状态
# ============================================================
def _get_task(task_id:str)-> Dict[str, Any] | None:
    """
    根据task_id找到任务信息。
    """
    with _task_lock:
        # get找不到就返回None
        task = _task_status.get(task_id)
        if task is None:
            return None
        # copy一份再返回
        # 防止外面直接修改原字典
        return task.copy()

# ============================================================
# 6.根据LangGraph的state判断大概执行到哪一步
# ============================================================
def _update_progress_from_state(task_id:str,state: Dict[str, Any])-> None:
    """
    LangGraph每执行完一个节点，
    stream模式都会给我们一个新的state。
    我们就根据state里已经有什么，
    大概判断现在执行到哪里了。
    """
    # --------------------------------------------------------
    # 已经生成向量
    # --------------------------------------------------------
    chunks = state.get("chunks") or []
    if chunks:
        first_chunk = chunks[0]
        # chunks里面已经有dense_vector
        # 说明BGE向量化已经做完
        if first_chunk.get("dense_vector"):
            _update_task(task_id,
            status = "processing",
            progress = 90,
            message = "向量生成完成，正在写入Milvus")
            return
        # chunks里面已经有item_name
        # 说明商品名识别基本做完
        if  "item_name" in first_chunk:
            _update_task(task_id,
                         status="processing",
                         progress=75,
                         message="商品名称识别完成，正在生成向量" )
            return
        # 有chunks但是还没向量
        # 说明文档已经切好了
        _update_task(task_id,
                     status="processing",
                     progress=60,
                     message="文档切片完成，正在识别商品名称" )
        return
    # --------------------------------------------------------
    # 已经拿到Markdown正文
    # --------------------------------------------------------
    if state.get("md_content"):
        _update_task(task_id,
                     status="processing",
                     progress=40,
                     message="文档解析完成，正在处理图片和切片" )
        return
    # --------------------------------------------------------
    # 已经判断完文件类型
    # --------------------------------------------------------
    if (state.get("is_pdf_read_enabled") or state.get("is_md_read_enabled")):
        _update_task(task_id,
                     status="processing",
                     progress=20,
                     message="文件检查完成，正在解析文档")
        return
# ============================================================
# 7.后台真正执行知识库导入
# ============================================================
def _run_import_task(task_id:str,import_file_path:str)-> None:
    """
    这个方法才是真正跑LangGraph的。
    文件已经上传完成以后：
    NodeEntry
    ↓
    PDF转MD
    ↓
    图片处理
    ↓
    文档切片
    ↓
    商品识别
    ↓
    BGE向量化
    ↓
    写入Milvus
    """
    try:
        # ----------------------------------------------------
        # 1.告诉前端任务开始执行
        # ----------------------------------------------------
        _update_task(
            task_id,
            status="processing",
            progress=10,
            message="开始导入知识库",
        )
        # ----------------------------------------------------
        # 2.创建一份新的LangGraph初始state
        # ----------------------------------------------------
        state = get_default_state()
        # 把当前任务ID放进去
        state["task_id"] = task_id
        # 把刚刚上传好的文件路径放进去
        state["import_file_path"] = import_file_path
        # ----------------------------------------------------
        # 3.创建导入工作流
        # ----------------------------------------------------
        workflow = KBImportWorkflow()
        # ----------------------------------------------------
        # 4.用stream模式执行
        # ----------------------------------------------------
        """
        普通invoke：
        workflow.run(state)
        只能最后一次性拿结果。
        stream=True：
        每执行完一个节点，
        都会给我们一次新的state。
        这样我们就可以顺手更新前端进度。
        """
        final_state = state
        for current_state in workflow.run(state,stream=True):
            final_state = current_state
            # 根据state判断现在大概执行到哪
            _update_progress_from_state(task_id, current_state)
        # ----------------------------------------------------
        # 5.工作流全部执行完成
        # ----------------------------------------------------
        chunks = final_state.get("chunks") or []
        # 整理一个比较简单的最终结果
        result = {
            "file_title":final_state.get("file_title"),
            "item_name": final_state.get("item_name"),
            "md_path": final_state.get("md_path"),
            "chunk_count": len(chunks),
        }
        # ----------------------------------------------------
        # 6.告诉前端：全部成功
        # ----------------------------------------------------
        _update_task(
            task_id,
            status="completed",
            progress=100,
            message="知识库导入完成",
            result=result,
            error=None
        )
    except Exception as e:
        # ----------------------------------------------------
        # 任何节点报错，都把错误保存下来
        # ----------------------------------------------------
        _update_task(
            task_id,
            status="error",
            message="知识库导入失败",
            error=str(e),
        )
# ============================================================
# 8.创建FastAPI
# ============================================================
app = FastAPI(
    title = "掌柜智库",
    description="知识库文件导入服务",
    version="1.0.0",
)
# ============================================================
# 9.允许前端访问这个后端
# ============================================================
app.add_middleware(
    CORSMiddleware,
    # 允许不同地址的前端访问
    allow_origins=["*"],
    # 当前项目暂时不用Cookie认证
    allow_credentials=False,
    # GET、POST等请求全部允许
    allow_methods=["*"],
    # 请求头全部允许
    allow_headers=["*"],
)
# ============================================================
# 10.挂载前端静态文件
# ============================================================
"""
以后我们创建：
knowledge_base/front/
里面放：
import.html
chat.html
就可以通过
/front/import.html
/front/chat.html
访问。
"""
if FRONT_DIR.exists():
    app.mount(
        "/front",
        StaticFiles(
            directory=str(FRONT_DIR)
        ),
        name="front",
    )
# ============================================================
# 11.测试后端有没有启动成功
# ============================================================
@app.get("/")
async def root():
    """
    浏览器访问：
    http://127.0.0.1:8000/
    如果看到JSON，
    就说明FastAPI启动成功。
    """
    return {
        "message": "掌柜智库导入服务运行正常"
    }
# ============================================================
# 12.打开导入页面
# ============================================================
@app.get("/import")
async def import_page():
    """
    浏览器访问：
    http://127.0.0.1:8000/import
    就返回front/import.html。
    """
    import_html = FRONT_DIR / "import.html"
    # 前端文件现在还没写的话
    # 就直接告诉我们文件不存在
    if not import_html.exists():
        raise HTTPException(
            status_code=404,
            detail="front/import.html还没有创建",
        )
    return FileResponse(path=str(import_html))
# ============================================================
# 13.上传PDF或者MD
# ============================================================
@app.post("/upload")
async def upload_file(
        background_task: BackgroundTasks,
        file: UploadFile = File(...),
):
    """
    前端上传文件以后：
    1.检查文件
    2.生成task_id
    3.保存文件
    4.后台开始跑LangGraph
    5.立刻把task_id返回给前端
    """
    # --------------------------------------------------------
    # 1.检查有没有文件名
    # --------------------------------------------------------
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="没有收到文件",
        )
    # --------------------------------------------------------
    # 2.只取真正的文件名
    # --------------------------------------------------------

    """
    比如前端传进来：
    C:/abc/test.pdf
    最后只拿：
    test.pdf
    防止路径乱跑。
    """
    safe_filename = Path(file.filename).name
    # --------------------------------------------------------
    # 3.检查后缀
    # --------------------------------------------------------
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="目前只支持PDF和MD文件",
        )
# --------------------------------------------------------
# 4.生成唯一任务ID
# --------------------------------------------------------
    task_id = str(uuid.uuid4())
    # --------------------------------------------------------
    # 5.每个任务建一个自己的文件夹
    # --------------------------------------------------------
    task_dir = (
        TEMP_UPLOAD_DIR
        / task_id
    )
    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    # --------------------------------------------------------
    # 6.最终文件保存位置
    # --------------------------------------------------------
    import_file_path = (
        task_dir
        / safe_filename
    )
    # --------------------------------------------------------
    # 7.把上传文件真正保存到硬盘
    # --------------------------------------------------------
    try:
        await file.seek(0)
        with open(import_file_path, "wb") as output_file:
            # 一边读取上传文件
            # 一边写入本地文件
            shutil.copyfileobj(file.file, output_file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件保存失败：{e}",
        )
    finally:
        await file.close()
    # --------------------------------------------------------
    # 8.先创建任务状态
    # --------------------------------------------------------
    _update_task(
        task_id,
        status="pending",
        progress=0,
        message="文件上传完成，等待知识库处理",
        result=None,
        error=None,
    )
    # --------------------------------------------------------
    # 9.把LangGraph放到后台执行
    # --------------------------------------------------------
    """
    为什么放后台？
    因为：
    PDF解析
    ↓
    BGE向量化
    ↓
    Milvus入库
    可能需要几十秒甚至更久。
    如果这里一直等，
    前端上传按钮就一直卡住。
    BackgroundTasks就是：
    先把task_id返回前端，
    然后后台慢慢跑导入流程。
    """
    background_task.add_task(
        _run_import_task,
        task_id=task_id,
        import_file_path=str(import_file_path),
    )
    # --------------------------------------------------------
    # 10.马上把task_id返回前端
    # --------------------------------------------------------
    return {
        "task_id": task_id,
        "message":"文件上传成功"
    }
# ============================================================
# 14.查询导入进度
# ============================================================
@app.get("/status/{task_id}")
async def get_task_status(task_id: str,):
    """
    前端会不断访问：
    GET /status/abc123
    后端就告诉它：
    现在进度多少
    是否完成
    有没有报错
    """
    # 根据task_id找任务
    task = _get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail="没有找到这个导入任务",
        )
    return task
# ============================================================
# 15.直接运行这个文件时启动服务器
# ============================================================
if __name__ == "__main__":
    # 开启项目日志
    setup_logging()
    # 启动FastAPI服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )

