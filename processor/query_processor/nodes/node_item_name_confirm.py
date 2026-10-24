#商品名称确认节点
#就是上下文管理机制，
#拿对话框id和原话，
#从mongoDB捞最近的聊天记录做成history，
#然后把用户这句话也存进去MongoDB得到messageid，
#最后把原话和历史喂给大模型，
#这样大模型知道的就更准确了
#最后输出提取后的准确信息，
#里面有商品名字和改写后的问题
#最后塞到state
import json
from utils.mongo_history_utils import update_message_item_names
from utils.milvus_utils import create_hybrid_search_requests,hybrid_search
from config.milvus_config import milvus_config
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config.lm_config import lm_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from typing import Dict,List,Tuple
from utils.mongo_history_utils import get_recent_messages, save_chat_message
from processor.query_processor.prompts.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE,ITEM_NAME_EXTRACT_SYSTEM_PROMPT
class NodeItemNameConfirm(NodeBase):
    name:str = "node_item_name_confirm"
#=======主流程====================
    def process(self,state:QueryGraphState)->QueryGraphState:
        """
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 步骤1 校验参数
        session_id, original_query = self._step_1_validate_param(state) #得到对话框id和用户原始问题
        logger.info(f"步骤1：参数校验通过")
        # 步骤2 获取历史记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到{len(history)}条历史消息")
        state["history"] = history #更新下状态
        # 步骤3 用户初始消息保存
        message_id = save_chat_message(session_id, "user",original_query)
        logger.info(f"步骤3：用户消息已初始保存，ID：{message_id}")
        # 步骤4 提取信息
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names",[])
        rewritten_query = extract_res.get("rewritten_query",original_query)
        #更新状态
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names
        #步骤5/6 如果有搜索到商品名字，进行搜索和对齐
        align_result = {} #先准备一个空字典保存后面的商品名对齐结果
        if len(item_names)>0:
            query_result = self._step_5_vectorize_and_query(item_names) #先把商品名向量化，然后去Milvus搜最相似的标准商品名
            align_result = self._step_6_align_item_names(query_result)
        else:
            logger.info("Node:未提取到商品名，跳过向量解锁")
        # 步骤7. 检查确认状态
        state = self._step_7_check_confirmation(state,align_result,history)
        # 步骤8. 写入最终历史
        state = self._step_8_write_history(state,session_id, rewritten_query,message_id)
        return state #把更新完的state交给下一个工作流节点
    """
    此时state结构：
    state = {
    "session_id": "s001",
    "original_query": "怎么调它的温度？",
    "history": [...],                                 # 最近10条聊天记录
    "rewritten_query": "HAK180烫金机怎么调温度？",        # 大模型改写后
    "item_names": ["HAK180烫金机"],                     # 确认后的标准商品名
    "answer": "",                                     # 没反问就留空
    }
    """
#=======做数据校验的方法，看下对话id和初始问题====================
    def _step_1_validate_param(self,state:QueryGraphState)-> Tuple:
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")
        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")
        return session_id, original_query
    #=======根据初始问题和历史提取信息的方法==========
    #把当前问题和历史记录丢给大模型，让他推理出
    #1.用户问的哪个商品item_name
    #2.补全它，这个后的完整问题rewritten_query
    #3.出错就兜底返回空商品名字+原问题
    def _step_4_extract_info(self,query:str,history:list)->Dict:

        """
        利用llm从当前问题及历史会话中提取出主要询问的商品名称item_names(可多个，JSON列表形式)
        若商品名字不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
        :param original_query: 字符串：用户原始查询问题
        :param history:列表[字典]：近期会话历史，每条消息含role/text字段，
        格式：[{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}, ...]
        :return: 字典-提取结果，固定包含两个字段，物品名字和提取信息：
                {"item_names": ["商品名1", "商品名2", ...],  # 提取的商品名列表，无则空列表
                "rewritten_query": "改写后的完整问题"       # 包含商品名的独立问题，无则返回原始query}
        """
        try:
            #1.获取llm客户端
            chat_model = ChatOpenAI(
                model=lm_config.item_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                #开启JSON标准输出模式，强制模型返回可解析的json_object
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            #2.构造历史对话文本，拼接为“角色:内容”格式，供llm做上下文理解
            history_text = "" #先构建一个字符串叫history_text
            for msg in history: #遍历搜索history里面的消息
                role = msg["role"]
                content = msg.get("text")
                history_text += f"{role}:{content}\n" #这样history_text就是有角色和文本了
            #3.处理和动态拼接提示词
            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=query
            )
            #4.构造LLM调用的消息列表，包含系统角色（定义助手身份）和用户角色（传入提示词）
            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)]
            #5.调用LLM客户端，发起请求获取结果
            response = chat_model.invoke(messages)
            content = response.content  #得到结果文本
            #6.数据清洗：处理LLM可能返回的代码块格式，（如```json ... ```），去除包裹符
            if content.startswith("```json"):
                content = content.replace("```json","").replace("```","")
            #7.数据解析：将json字符串转为字典
            result = json.loads(content)
            #8.健壮性处理：确保字段存在
            if "item_names" not in result:
                result["item_names"] = []
            if "rewritten_query" not in result:
                result["rewritten_query"] = query
            #9.给item_names去除空格
            result["item_names"] = [
            name.replace("\n","").replace("\t","").replace("\r","")
            for name in result["item_names"]]
            #10.返回解析后的提取结果
            return result
        except Exception as e:
            logger.error(f"大模型调用异常：{e}")
            return {"item_names": [], "rewritten_query":query}
# =======商品名字拿去向量库里搜一遍，找回来候选+相似度分数==========
    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """
        把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
        :param item_names:列表[字符串] - 步骤4中 提取的商品名列表（如["苹果15", "华为P60"]）
        :return:列表[字典] - 格式：
        [{"extracted_name": "提取的原始商品名",  # 如"苹果15"
         "matches": [ # 该商品名的TopN匹配结果，无则空列表
         {  "item_name": "数据库中的商品名",  # Milvus中存储的标准化商品名
            "score": 0.98  },   ]      },       # 混合搜索的相似度评分（0-1，越高越相似）
        """
        results = []
        if not item_names:
            return results
        client = get_milvus_client()
        if not client:
            logger.error("连接error失败")
            return results
        collection_name = milvus_config.item_name_collection #把之前调用大模型生成的物品名字集合拿出来
        if not collection_name or not client.has_collection(collection_name):
            logger.warning(
                f"Milvus集合[{collection_name}]不存在，跳过商品名对齐，继续无商品名过滤检索"
            )
            return [
                {"extracted_name": item_name, "matches": []}
                for item_name in item_names
            ]
        embeddings = generate_embeddings(item_names) #生成现在用户问题的物品名字的切片
        for i in range(len(item_names)):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]
                reqs = create_hybrid_search_requests(
                    dense_vector = dense_vector,
                    sparse_vector = sparse_vector,
                    limit = 5
                )
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8,0.2),
                    limit = 5,
                    norm_score=True,
                    output_fields=["item_name"]
                )
                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        matches.append(
                            {
                                "item_name": hit.get("entity",{}).get("item_name"),
                                "score": hit.get("distance"),
                            })
                    #search_res是列表有字典，对应[{"item_name":"xxxx","score":95}]
                results.append({"extracted_name": item_names[i], "matches": matches})
                """
                最终results格式：
                results = [
        {
            "extracted_name": "苹果15",              # 用户说的（不标准）
            "matches": [                             # 库里匹配到的候选
                {"item_name": "苹果iPhone15",  "score": 0.95},
                {"item_name": "苹果iPhone14",  "score": 0.82},
            ]
        },
        {
            "extracted_name": "华为P60",
            "matches": [
                {"item_name": "华为P60 Art",  "score": 0.91},
                {"item_name": "华为P60 Pro",  "score": 0.88},
            ]
        },
    ]
                """
            except Exception as e:
                logger.error(f"查询商品名字{item_names[i]}时候出错")
        return results
    # =======按分数分拣商品名，输出 {"confirmed_item_names": [...], "options": [...]}==========
    def _step_6_align_item_names(self, query_results) -> dict:
        """
        根据Milvus搜索评分，逐个对齐step4提取的item_names，生成【确认商品名】和【候选商品名】
        对齐规则（优先级a>b>c>d）:
        a  如果只有一个匹配结果评分高于0.85 → 直接确认该商品名
        b  如果多条匹配结果评分超过0.85 → 优先取与原始提取名相同的，无则取分数最高的
        c  如果无0.85分以上结果 → 取分数≥0.6的最高前5个作为候选
        d  如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
        :param  query_results:列表[字典]，step5的返回结果
        :return:字典
        """
        confirmed_item_names :List[str] = [] #
        options: List[str] = [] #
        logger.info(f"步骤6：获得待处理的数据源：{query_results}")
        for res in query_results:
            extracted_name = (res.get("extracted_name","") or "").strip()
            matches = res.get("matches",[]) or []
            if not matches:
                continue
            high = [m for m in matches if m.get("score",0)>0.85]  #把分数大于0.85的全部挑出来当高置信度结果
            mid = [m for m in matches if m.get("score",0)>0.6] #把分数大于等于0.6的全部挑出来当中等置信度结果
            if len(high)>0:
                for m in high:
                    confirmed_item_names.append(m.get("item_name"))
                continue
            if len(mid)>0:
                for m in mid[:3]: #只拿前3个中等分数结果
                    options.append(m.get("item_name"))
        return {
                "confirmed_item_names": list(set(confirmed_item_names)), #把直接确认的商品名去重以后返回
                "options": list(set(options)), #把候选商品名去重以后返回
            }
    # =======根据步骤6分3个分支更新state，并同步更新历史消息的商品名关联==========
    def _step_7_check_confirmation(self,state,align_result,history):
        """
        :param state: 字典 - 原始会话状态，包含session_id/original_query等核心字段
        :param align_result: 字典 - step6的对齐结果
        :param history: 列表[字典] - 近期会话历史
        :return: 字典 - 更新后的会话状态，包含item_names/answer
        """
        confirmed = align_result.get("confirmed_item_names",[]) #拿可以直接确认的商品名列表
        options = align_result.get("options",[])  #拿需要用户选择的候选商品名列表
        if confirmed: #情况1：已经有高置信度商品名，可以直接确认
            ids_to_updata = [] #准备一个列表保存之前那些还没有绑定商品名的历史消息id
            for msg in history: #遍历之前的历史聊天
                if not msg.get("item_name") : #如果这条旧消息还没有商品名
                    mid = msg.get("_id") #拿这条MongoDB消息自己的id
                    if mid: #如果这个id确实存在
                        ids_to_updata.append(str(mid)) #转成字符串以后加入待更新列表
            if ids_to_updata: #如果真的有需要补商品名的历史消息
                update_message_item_names(ids_to_updata,confirmed)  #把这些历史消息统一补上现在确认出来的商品名
            state["item_names"] = confirmed #把最终确认商品名写进state
            state["answer"] = "" #已经确认商品了，不需要在这里问用户，所以answer留空
            return state
        if options: #情况2：没有高置信度商品，但是有几个可能的候选商品
            options_str = ",".join(options) #把候选商品拼成"商品A、商品B、商品C"
            answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。" #生成一句让用户确认型号的话
            state["answer"] = answer #把这句话放进state，后面直接回答用户
            state["item_names"] = []  #因为现在还没真正确认，所以商品名先清空
            return state
        # 情况3：未提取到商品名或 Milvus 没有匹配项，保留无过滤的检索流程。
        state["answer"] = ""
        state["item_names"] = []
        return state

    def _step_8_write_history(self,state,session_id, rewritten_query,message_id):#把这次最终处理出来的信息重新保存进MongoDB
        """
        把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
        包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
        :param state:  step6更新后的会话状态，包含answer/item_names等字段
        :param session_id: 字符串 - 会话唯一标识
        :param rewritten_query: 字符串 - step3改写后的完整问题
        :param message_id: 字符串 - 本次用户问题的消息唯一ID
        :return:
        """
        if state.get("answer"):  #如果step7生成了需要直接回复用户的话
            save_chat_message(
                session_id = session_id, #属于当前这个对话窗口
                role = "assistant", #这条消息是助手说的
                text = state["answer"], #具体保存step7生成的回答
                rewritten_query = "", #助手消息不需要重写问题，所以这里空字符串
                item_names = state.get("item_names",[]), #同时保存这条消息关联的商品名，没有就是空列表
                )
        save_chat_message(  #再把用户刚才那条已经初步保存过的消息更新一次
            session_id = session_id,  # 还是当前对话窗口
            role = "user",  # 这条消息是用户的
            text = state["original_query"],  # 用户最原始的问题
            rewritten_query=rewritten_query,  # 现在把大模型重写后的完整问题补进去
            item_names = state.get("item_names", []),  # 同时保存这条消息关联的商品名，没有就是空列表
            message_id = message_id )#指定之前那条用户消息id，所以这里是更新旧消息，不是新增一条
        return state
if __name__ == "__main__":

    init_state = {
        "session_id": "test_session_002", #当前测试对话窗口的id
        "original_query": "怎么调他的转印温度？" #用户原始问题
    }

    node_item_name_confirm = NodeItemNameConfirm() #实例化商品名称确认节点
    result = node_item_name_confirm(init_state) #执行节点测试
    logger.info(format_json(result)) #打印最终state
