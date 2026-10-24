from typing import Tuple,Dict,List
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import DataType
from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import LLMError, MilvusError, StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client ,escape_milvus_string
from processor.query_processor.prompts.item_name_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT

"""
这里我开始混乱了，所以举个具体例子：
state = {
    "file_title":"华为Mate80使用说明书",
    "md_content":"这里是整篇md正文",
    "chunks":[
        {
            "title":"#产品介绍",
            "content":"华为Mate80是一个xxxx",
            "file_title":"华为Mate80使用说明书"
        },
        {
            "title": "# 使用方法",
            "content": "长按电源键开机……",
            "file_title": "华为Mate80使用说明书"
        }
    ]
}
"""
class NodeItemNameRecognition(BaseNode): #读一段文本，然后判断/提取这段内容应该叫什么名字。
    name:str = "node_item_name_recognition"
    def process(self,state:ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：商品主体名称识别
        流程总览：
        1.先拿到task_id、文件名和切片内容
        2.把这些内容整理成大模型能看懂的的输入
        3.让大模型判断这份内容对应的商品名称
        4.把识别出来的商品名称写回状态和每个切片
        5.把商品名称转成dense和sparse两种向量
        6.最后把这些数据存进Milvus向量数据库
        :param state:工作流状态对象
        :return:更新后的状态对象
        """
        #这里写一个整体流程
        #先拿 file_title + chunks
        # 取前几个 chunks 拼成 context
        # 大模型根据 file_title + context 识别出一个单独的 item_name
        # 把这个 item_name 写回 state 和所有 chunk
        # 再给这个 item_name 生成稠密向量和稀疏向量
        # 最后把 file_title + item_name + 两种向量 存进 Milvus。
        file_title,chunks  = self._step_1_getinputs(state) #先拿到file_name和chunks就是state里的文件名字和切片内容
        context = self._step_2_build_context(chunks) #2.把前几个chunks拼成一个长的context字符串
        item_name = self._step_3_call_llm(file_title,context) #3.调用大模型把文件名字file_name和大字符串context输入进去得到物品名字，从这里开始输出就是单个了，前面都是整体
        self._step_4_update_chunks(state,chunks,item_name)  #4.把商品名称写回state和chunks，这里是给chunks加个新字段item_name,同时state也更新一下item_name，后面有state新实例
        dense_vector,sparse_vector = self._step_5_generate_vectors(item_name)  #这里是得到稠密向量和稀疏向量，稠密是信息多适合模糊对应，稀疏是关键词对应
        self._step_6_save_to_milvus(state,file_title,item_name,dense_vector,sparse_vector) #把之前得到的东西传给milvus，状态，文件名，物品名字，稠密向量和稀疏向量
        self.logger.info(f"---识别成功：{item_name}")
        return state
    def _step_1_getinputs(self,state:ImportGraphState) -> Tuple[str,List[Dict[str,str]]]:
        # 传进去state，返回一个元组：第一个是字符串file_title;第二个是由很多字典组成的列表chunks
        file_title = state.get("file_title") #拿文件标题
        if not file_title: #没有就报错
            raise StateFieldError(
                field_name="file_title",
                message="文件标题不能为空",
                expected_type=str
            )
        #拿chunks
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(
                field_name="chunks",
                message="chunk不能为空",
                expected_type=list
            )
        #检查chunks必须是list
        if not isinstance(chunks,list):
            raise StateFieldError(
                field_name="chunks",
                message="chunks数据类型不正确",
                expected_type=list
            )
        return file_title,chunks
    def _step_2_build_context(self,chunks:List[Dict[str,str]],)-> str: #传进去state，得到上下文context
        parts:list[str] = [] #造一个parts列表，里面是str，存每个切片文本
        total_chars = 0 #记录目前一共多少字符
        for idx,chunk in enumerate(chunks[:self.config.item_name_chunk_k],start=1): #每次只拿chunks的前5个，然后编号1-5
            chunk_title = chunk.get("title","").strip() #拿标题
            chunk_content = chunk.get("content","").strip() #拿正文
            piece = ( #这里把切出来的前五个都放成piece
                f"【切片{idx}】\n"
                f"【标题{chunk_title}】\n"
                f"【内容{chunk_content}】\n"
            )
            parts.append(piece) #加到之前创建的字符串列表parts里
            total_chars += len(piece) #计算下一共多少字符
            if total_chars > self.config.item_name_chunk_size: # 要是字符串太长了就警告然后停止for循环不再遍历
                self.logger.warning(
                    f"累计字符数{total_chars}"
                    f"已超过限{self.config.item_name_chunk_size}"
                    f"停止切分"
                )
                break
        # 把parts列表里的所有字符串用两个换行符连接起来，
        # 拼成一个大的context字符串，并去掉首尾多余的空白字符
        context = "\n\n".join(parts).strip()
        final_context = context[:self.config.item_name_chunk_size]
        return final_context
    def _step_3_call_llm(self,file_title:str,context:str) -> str:
        if not context:
            raise LLMError("没有可用于识别商品名的文档内容", node_name=self.name)
        try:
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                file_title=file_title,
                context=context,
            )
            llm = ChatOpenAI(
                model = lm_config.item_model or lm_config.llm_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                extra_body={
                    "enable_thinking":False,#不开深度思考模式
                }
            )
            messages = [
                SystemMessage(
                    content=ITEM_NAME_SYSTEM_PROMPT,
                ),
                HumanMessage(
                    content=user_prompt
                )
            ]
            response = llm.invoke(messages) #调用大模型
            item_name = response.content #拿到物品名字
            item_name = item_name.replace( #这里是去除一些字符
                "\n","").replace("\r","").replace("\t","").replace(" ","")
            if not item_name:
                raise LLMError("大模型没有返回商品名称", node_name=self.name)
            return item_name
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(
                "商品名称识别的大模型调用失败",
                node_name=self.name,
                cause=e,
            ) from e
    def _step_4_update_chunks(self,state:ImportGraphState,chunks:List[Dict[str,str]],item_name:str)->None:
        for chunk in chunks:
            chunk["item_name"] = item_name #这里就是给每个chunk加上对应的chunk，从方法3开始就从前面的一大串变成单个字符串item_name了
        state["chunks"] = chunks #这里就是state状态更新
        state["item_name"] = item_name #state自己也保存item_name
        """
            此时state状态变成：
            state = {
    "file_title": "华为Mate80使用说明书",
    "md_content": "这里是整篇Markdown正文……",
    "chunks": [
        {
            "title": "产品介绍",
            "content": "华为Mate80是一款手机……",
            "item_name": "华为Mate80"
        },
        {
            "title": "开机方法",
            "content": "长按电源键即可开机……",
            "item_name": "华为Mate80"
        },
        {
            "title": "充电方法",
            "content": "连接充电器即可充电……",
            "item_name": "华为Mate80"
        }
    ],
    "item_name": "华为Mate80"
    }"""
    def _step_5_generate_vectors(self,item_name:str) -> Tuple[List,Dict]:
        if not item_name: #商品名称为空就没有向量
            raise StateFieldError(
                field_name="item_name",
                message="商品名称不能为空，无法生成向量",
                expected_type=str,
            )
        vectors = generate_embeddings([item_name]) #这里得到了一个vectors字典，里面是稠密向量键值对和稀疏向量键值对
        return (
            vectors["dense"][0],
            vectors["sparse"][0]   #因为是只传进去一条，所以拿第一条信息
        )
    def _step_6_save_to_milvus(self,state,file_title,item_name,dense_vector,sparse_vector)-> None:
        try:
            milvus_client = get_milvus_client() #拿Milvus客户端，之前那个utils方法里有写
            if not milvus_client:
                raise MilvusError("无法获取Milvus客户端", node_name=self.name)
            #商品名称合集，这里就是先给一个空集合得到milvus配置里面那个集合，空集合取名字叫collection_name
            collection_name = (milvus_config.item_name_collection)
            if not collection_name:
                raise MilvusError("商品名称集合配置为空", node_name=self.name)
            if not milvus_client.has_collection(collection_name): #没有集合就创建
                self._create_item_name_collection(
                    collection_name,
                    milvus_client,
                    len(dense_vector),
                )
            safe_item_name = escape_milvus_string(item_name) #把字符串里的特殊字符处理一下，避免后面拼Milvus查询条件时报错，在milvus里面有写
            filter_expr = (f'item_name=="{safe_item_name}"') #这里就是拼接变量名字， filter_expr以后都是'item_name=="华为Mate80"'这种形式，方便后面删除
            milvus_client.delete(collection_name=collection_name, filter=filter_expr) #这个delete是milvus自带的方法，里面就传两个参数collection_name和filter
            #到这里都是先写好删除重复的方法，后面才是真正存进去
            data = {"file_title": file_title,"item_name": item_name} #data先把file_title和item_name存进来
            if dense_vector is not None: data["dense_vector"] = dense_vector #稠密向量有就加进来，它是个列表
            if sparse_vector is not None: data["sparse_vector"] = sparse_vector #稀疏向量有就加进来，它是个字典
            result = milvus_client.insert(collection_name=collection_name, data=[data]) #Milvus批量接口，即使一条也要放进列表
            if result.get("insert_count") != 1:
                raise MilvusError(
                    f"商品名称写入数量异常：{result}",
                    node_name=self.name,
                )
            milvus_client.flush(collection_name=collection_name)
            milvus_client.load_collection(collection_name=collection_name)
            state["item_name"] = item_name
        except Exception as e:
            if isinstance(e, MilvusError):
                raise
            raise MilvusError(
                "商品名称写入Milvus失败",
                node_name=self.name,
                cause=e,
            ) from e
    def _create_item_name_collection(self,collection_name,milvus_client,vector_dimension): #这里就是看有无那个集合collection，没有的要用新方法创建
         schema = milvus_client.create_schema(auto_id=True,enable_dynamic_field=False) #schema就是规定这个表有啥有啥，每个字段是啥类型的id自动赋值
         # 主键
         schema.add_field(field_name="pk",datatype=DataType.INT64,is_primary=True,auto_id=True)
         # 文件标题
         schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=512)
         # 商品名称
         schema.add_field(field_name="item_name", datatype=DataType.VARCHAR,max_length=512)
         # dense向量
         schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=vector_dimension)
         # sparse向量
         schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
         # 创建索引配置
         index_params = (milvus_client.prepare_index_params())
         # dense索引
         index_params.add_index(field_name="dense_vector",index_name="dense_vector_index",index_type="AUTOINDEX",metric_type="COSINE")
         # sparse索引
         index_params.add_index(field_name="sparse_vector",index_name="sparse_vector_index",index_type="SPARSE_INVERTED_INDEX",
         metric_type="IP",params={"inverted_index_algo": "DAAT_MAXSCORE","normalize": True,"quantization": "none"})
         # 真正创建集合
         milvus_client.create_collection(collection_name=collection_name,schema=schema,index_params=index_params)







