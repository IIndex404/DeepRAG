"""
答案输出
"""
from typing import Dict, List, Tuple
from processor.query_processor.base import NodeBase
from processor.query_processor.prompts.answer_prompt import ANSWER_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.llm_utils import get_llm_client
from utils.mongo_history_utils import save_chat_message
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import set_task_result
import re
MAX_CONTEXT_CHARS=12000
class NodeAnswerOutput(NodeBase):
    name:str = "node_answer_output"
    def process(self, state:QueryGraphState)->QueryGraphState:
        """
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        logger.info(f"【{self.name}】开始生成最终答案")
        # 图片只允许从检索到的本地切片中提取，不让大模型自行生成图片地址。
        image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])
        state["image_urls"] = image_urls
        if state.get("answer"):
            state["answer"] = self._strip_model_image_block(state["answer"])
        # =====================================================
        # Step1：先看看前面的节点是不是已经直接给出了答案
        # =====================================================
        answer_exists = self._step_1_check_answer(state)
        # =====================================================
        # 2.没有现成答案，就拼Prompt
        # =====================================================
        if not answer_exists:
            prompt = self._step_2_construct_prompt(state)
            state["prompt"] = prompt #把最终Prompt存回state
            # 3.调用大模型生成答案
            self._step_3_generate_response(state,prompt)
            state["answer"] = self._strip_model_image_block(state.get("answer") or "")
        # 非流式任务只保存过滤后的最终答案。
        if state.get("answer") and not state.get("is_stream"):
            set_task_result(state["session_id"],"answer",state["answer"])
        # 4.有答案就保存到MongoDB历史记录
        if state.get("answer"):
            logger.info("---写入MongoDB历史记录---")
            self._step_4_write_history(state,image_urls=image_urls)
        # 5.如果是流式输出，告诉前端回答结束了
        if state.get("is_stream"):
            push_to_session(
                state["session_id"],
                SSEEvent.FINAL,
                {
                    "answer":state["answer"],
                    "status":"completed",
                    "image_urls":image_urls,
            })
        logger.info("---node_answer_output节点处理结束---")
        return state
    # Step1：检查是不是已经有答案
    def _step_1_check_answer(self,state: QueryGraphState) -> bool:
        answer = state.get("answer") # 拿已有答案
        is_stream = state.get("is_stream")  # 看当前是不是流式输出
        if answer:  # 如果前面的节点已经给出了答案
            if is_stream: # 如果是流式输出
                logger.info("---Step1：发现已有答案，直接流式推送---")
                push_to_session(state["session_id"],SSEEvent.DELTA,{"delta":answer})
            else:
                logger.info("---Step1：发现已有答案，直接使用---")
            return True
        return False
    # Step2：现拼Prompt存进去state
    def _step_2_construct_prompt(self,state: QueryGraphState) -> str:
        char_budget = MAX_CONTEXT_CHARS  #一共最多给参考内容12000个字符
        # 1.拿用户问题，优先用改写后的
        question = (state.get("rewritten_query") or state.get("original_query","") )
        # 2.拿商品名/实体名
        item_names = state.get("item_names") or []
        # 3.把Rerank后的文档整理成参考内容
        context_str,char_budget = self._format_reranked_docs(state.get("reranked_docs") or [],char_budget)
        # 4.把历史聊天整理成字符串
        history_str, char_budget = self._format_chat_history(state.get("history") or [],char_budget)
        # 5.把item_names列表拼成字符串
        item_names_str = (",".join(item_names) if item_names else "无指定商品")
        # 6.把所有内容填进提示词模板
        prompt = ANSWER_PROMPT.format(
            question = question,
            history = history_str or "暂无历史对话",
            context = context_str or "无参考内容",
            item_names = item_names_str)
        logger.info( f"组装后的提示词为：{prompt}")
        return prompt
    def _format_reranked_docs(self,reranked_docs:List[Dict],char_budget:int)->Tuple[str,int]: #返回context_str,char_budget-used_chars)
        formatted_lines = [] # 存整理好的每篇文档
        used_chars = 0  # 已经用了多少字符
        for idx,doc in enumerate(reranked_docs,start=1):
            content = doc.get("content") or ""  # 拿文档正文
            meta_tags = [f"[{idx}]"]
            for field, template in [ #  把来源、chunk_id、url、标题也加进去
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = doc.get(field)
                if field_value is not None:
                    field_value = str(field_value).strip()
                    if field_value:
                        meta_tags.append(template.format(field_value))
                #规范化后：[1] [source=local] [chunk_id=c1] [title=温度设置] [score=0.9500] 温度调到110度
            relevance_score = doc.get("score") # 加上Rerank相关性分数
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")
            doc_entry = (
                " ".join(meta_tags)
                +"\n"
                + content
            )
            if used_chars + len(doc_entry) > char_budget: # 超过字符上限就停止继续加
                break
            formatted_lines.append(doc_entry) # 加进最终参考资料
            used_chars += len(doc_entry)+2 # 更新已经使用的字符数 +2那两个换行符（\n\n）的长度。
        context_str = "\n\n".join(formatted_lines)
        return (context_str,char_budget-used_chars)
    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """
        chat_history简单例子是：
        {"role": "user", "text": "你好"},
        {"role": "assistant", "text": "你好，请问有什么可以帮你？"},
        {"role": "user", "text": "这个机器怎么调温度？"},
        {"role": "assistant", "text": "请问是什么型号？"},
        {"role": "user", "text": "HAK180烫金机"},]
        """
        formatted_lines = []  # 存整理好的聊天记录
        used_chars = 0 # 已经用了多少字符
        role_label_map = {"user":"用户","assistant":"助手"} # 把角色名换成中文
        for message in chat_history:
            role = message.get("role")
            text = message.get("text")
            if not text or role not in role_label_map:  # 没内容或者角色不对就跳过
                continue
            formatted_line=(f"{role_label_map[role]}: {text}") ## 拼成：用户: xxx / 助手: xxx
            new_used_chars=(used_chars+len(formatted_line)+1)
            if new_used_chars > char_budget: break # # 超过剩余字符数就停止
            formatted_lines.append(formatted_line) # 加进历史聊天
            used_chars = new_used_chars
        history_str = "\n".join(formatted_lines)
        return (history_str,char_budget-used_chars)
    def _step_3_generate_response(self,state: QueryGraphState,prompt:str)->QueryGraphState: #调大模型生成最终答案,支持流式和非流式
        logger.info("---Step3：开始生成回答---")
        # 1.拿统一配置好的大模型客户端
        llm = get_llm_client()
        # 2.拿当前会话id
        session_id = state.get("session_id")
        # 3.看是不是流式输出
        is_stream = state.get("is_stream")
        # =====================================================
        # 流式输出
        # =====================================================
        if is_stream:
            final_text = "" # 用来一点点拼完整答案
            pending_text = "" # 暂存少量字符，防止【图片】标记被拆到多个分片
            image_block_started = False
            image_marker = "【图片】"
            try:
                for chunk in llm.stream(prompt): # 大模型一小段一小段生成
                    delta = (getattr(chunk,"content","")or "")
                    if not delta:
                        continue
                    final_text += delta # 拼到完整答案里
                    if image_block_started:
                        continue # 图片区块及其后续内容不推给前端
                    pending_text += delta
                    marker_pos = pending_text.find(image_marker)
                    if marker_pos >= 0:
                        safe_delta = pending_text[:marker_pos]
                        if safe_delta:
                            push_to_session(session_id,SSEEvent.DELTA,{"delta": safe_delta})
                        pending_text = ""
                        image_block_started = True
                        continue
                    keep_chars = len(image_marker) - 1
                    if len(pending_text) > keep_chars:
                        safe_delta = pending_text[:-keep_chars]
                        pending_text = pending_text[-keep_chars:]
                        push_to_session(session_id,SSEEvent.DELTA,{"delta": safe_delta})
                if pending_text and not image_block_started:
                    push_to_session(session_id,SSEEvent.DELTA,{"delta": pending_text})
                logger.info(f"流式输出完成，总长度：{len(final_text)}")
            except Exception as e:
                logger.error(f"流式生成出错：{e}",exc_info=True)
                push_to_session(session_id,SSEEvent.ERROR,{"error": str(e)})   # 出错也推给前端
            state["answer"] = self._strip_model_image_block(final_text)  # 只保存可信正文
        # =====================================================
        # 非流式输出
        # =====================================================
        else:
            try:
                response = llm.invoke(prompt) # 一次性生成完整答案
                content = response.content   # 拿回答正文
                state["answer"] = content # 存回state

                logger.info(f"生成回答完成，长度：{len(content)}")
            except Exception as e:
                logger.error(f"生成回答出错：{e}",exc_info=True)
                state["answer"] = ( "抱歉，生成回答时出现错误。")  # 兜底答案
        return state
    def _strip_model_image_block(self, answer: str) -> str:
        """移除模型生成的图片区块；可信图片由 _extract_images_from_docs 单独提供。"""
        answer = answer or ""
        cleaned = re.sub(r"\n*\s*【图片】[\s\S]*$", "", answer).rstrip()
        if cleaned != answer.rstrip():
            logger.warning("检测到模型输出的图片区块，已移除并改用本地切片图片")
        return cleaned
    def _extract_images_from_docs(self, docs: List[Dict])-> List[str]:
        """
        从参考文档里提取图片URL
        """
        images = [] # 最后所有图片放这里
        seen = set()  # 用来去重
        if not docs: return [] # 没有文档直接返回空列表
        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')  # 匹配Markdown图片：![图片说明](图片URL)
        for doc in docs:
            # 1.先检查文档自己的url字段
            url = (doc.get("url") or "").strip()
            if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")):
            # 如果这个url本身就是图片
                if url not in seen:  # 没出现过才加入
                    seen.add(url)
                    images.append(url)
            # 2.再检查正文里有没有Markdown图片
            content = (doc.get("content") or "").strip()
            if content:
                matches = md_img_pattern.findall(content) # 把正文里所有图片URL找出来
                for img_url in matches:
                    img_url = img_url.strip()
                    if (img_url and img_url not in seen):
                        seen.add(img_url)
                        images.append(img_url)
        return images
    def _step_4_write_history(self,state: QueryGraphState,image_urls:List[str]=None)-> QueryGraphState:
        # 1.拿当前会话id
        session_id = state.get("session_id","default")
        # 2.拿AI最终回答
        answer = (state.get("answer") or "").strip()
        # 3.拿这次对应的商品名
        item_names = (state.get("item_names") or [])
        try:
            if answer:
                save_chat_message(
                    session_id = session_id,
                    role = "assistant",
                    text = answer,
                    rewritten_query = "",
                    item_names = item_names,
                    image_urls = image_urls,
                    message_id = None,
                )
        except Exception as e:
            logger.error( f"写入Mongo历史记录失败: {e}")

        return state