import json
import logging
import re
from os.path import split
from pathlib import Path
from sys import exc_info
from typing import List,Tuple,Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt.tool_node import msg_content_output
from langgraph.pregel.debug import map_debug_tasks

from processor.import_processor.base import BaseNode,setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
class NodeDocumentSplit(BaseNode):
    """
    节点：文档切分
    将长文档切分成小的chunks以便检索
    """
    #覆盖基类的name属性，表示节点名称
    name:str = "node_document_split"
    def process(self,state:ImportGraphState)->ImportGraphState:
        """
        节点：文档切分（node_doecument_split）
        整体流程：加载输入——>按MD标题初步切-——>长切短合-——>统计输出-——>结果备份
        核心目的：将长MD文档切分成长度适中的chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各个步骤新增chunk元信息补充、自定义切分规则、向量入库前置处理等
        必要处理：md_content、file_title
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        #步骤1：加载并标准化输入数据
        content,file_title = self._self_1_get_inputs(state)
        #步骤2：按MD标题进行初次切分
        sections,title_count,lines_count = self._step_2_split_by_titles(content,file_title)
        #步骤3： 无标题场景兜底处理
        sections = self._ste_3_handle_no_title(content,sections,title_count,file_title)
        #步骤4： Chunk精细化处理（长切短合）
        sections = self._step_4_refine_chunks(sections)
        #步骤5： 输出文档切分统计信息
        self._step_5_print_stats(lines_count,sections)
        #步骤6： Chunk结果本地JSON备份
        self._step_6_backup(state,sections)
        #写入状态字典
        state["chunks"] = sections
        return state
    def _self_1_get_inputs(self,state:ImportGraphState)->Tuple[str,str]:
        """
        【步骤1】：获取并处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state:项目状态字典 包含md_content等核心键
        :return:标准化后的MD内容/文件标题
        """
        #1、非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title",message="文件标题不能是空",expected_type=str)
        #2、基础标准化：统一换行符
        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(
                field_name="md_content",
                expected_type=str,
            )
        md_content = md_content.replace("\r\n","\n").replace("\r","\n")
        return md_content,file_title
    def _step_2_split_by_titles(self,content:str,file_title:str)->Tuple[List[Dict[str,str]],int,int]:
        #这里给一个输出示例：[{"name": "图片1.png", "summary": "这是一张系统流程图",},{"name": "图片2.jpg","summary": "这是一张土地覆盖变化图",}])
        """
        【步骤2】按md标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理，将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的md完整内容
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """
        #1、定义标题正则
        #正则匹配md文件的 1-6级标题（核心规则，适配缩进/标准格式）
        #^\s*:行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}:匹配1-6个#（对应MD的1-6级标题）
        # \s+:#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+: 标题文字至少1个字符（避免空标题）
        title_pattern = r'^\s*#{1,6}\s+.+'
        #2、初始化需要的数据
        lines = content.split("\n")
        sections = [] #章节列表
        title_count = 0 #标题数量
        current_title = "" #当前的标题
        current_lines = [] #当前标题和下一个标题之间的文本内容
        in_code_block = False #代码标记快：False当前没在代码块中，True表示在
        #3、定义内部函数组装sections列表
        def _flush_sections():
            #内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过
            if not current_lines:
                return
            sections.append({
                "title":current_title,
                "content":"\n".join(current_lines),
                "file_title":file_title, # 记录当前章节所属的文件标题，方便后面知道这段内容来自哪个文件
            })
        #4、逐行遍历，识别标题和普通行以及代码块
        for line in lines:
            stripped_line = line.strip() # 去掉当前行开头和结尾的空格、Tab 等空白字符，
            # 4.1 识别代码快边界 ``` 或 ~~~
            if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
                in_code_block = not in_code_block  # 切换代码块状态：进入时变为 True，离开时变为 False
                current_lines.append(line)
                continue
            # 4.2 识别标题
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)# 使用前面定义的 title_pattern 正则规则，检查当前行 line 是否为 Markdown 标题
            if is_valid_title:
                ## 遇到新标题，先把上一章保存到 sections
                _flush_sections()
                # 保存当前读到的新标题
                current_title = stripped_line
                # 上一章已经结束，重新创建一个列表，
                # 并把新标题作为新章节的第一行
                current_lines = [current_title]
                title_count += 1
                self.logger.info(f"识别标题：{current_title}")
            else:
                #普通行
                current_lines.append(line)
        _flush_sections()
        self.logger.info(f"文档粗切（按标题切分）完成，共{len(sections)}个章节，标题数量是{title_count},文本共有{len(lines)}行")
        return sections,title_count,len(lines)
    def _ste_3_handle_no_title(self,content:str,sections:List[Dict[str,str]],title_count:int,file_title:str)->List[Dict[str,str]]:
        """
        【步骤3】无标题兜底处理
        功能：若MD未识别出到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content:标准化后的MD完整内容
        :param sections:步骤2切分后的章节列表
        :param title_count:步骤2识别的有效标题数量
        :param file_title:所属文件标题
        :return:兜底后的章节列表
        """
        if title_count == 0:
            #无标题情况，替换为单章节，标题为“无标题”
            self.logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
            return [{"title":"无标题","content":content,"file_title":file_title}]
        #有标题情况：直接返回步骤2的结果
        self.logger.debug(f"步骤3：检测到{title_count}个有效标题，无需兜底处理")
        return sections
    def _step_4_refine_chunks(self,sections:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
        执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvyus向量库schema）
        :param sections:步骤3处理后的章节列表
        :return:长度适中、低碎片化的最终chunk列表
        """
        #阶段1：切分超长章节 ——》 所有章节长度控制在最大长度内
        refined_split = []  # 用来装后面切好的章节
        for sec in sections: #逐个处理章节，切完的结果直接放到列表里
            refined_split.extend(self._split_long_section(sec))
        self.logger.info(f"步骤4-1：超长章节切分完成·，共生成{len(refined_split)}个初始子chunk")
        #阶段2：合并过短章节——》减少碎片化，提升后续检索/大模型调用效果
        final_sections = self._merge_short_section(refined_split)
        self.logger.info(f"步骤4-2：过短章节合并完成：最终得到{len(final_sections)}个Chunk")
        #阶段3：父标题兜底 -》适配Milvus向量库schema（parent——title为必填字段）
        #兜底规则：无parent_title则用自身title，title也无则填空字符串
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""
        self.logger.debug(f"步骤4-3：父标题兜底完成，所有chunk均包含parent_title字段")
        return final_sections
    def _split_long_section(self,section:Dict[str,str]) -> List[Dict[str,str]]: #把太长的章节继续切成更小的块，方便后面处理
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按【段落-》句子-》空格】从粗到细切分，保留语义
        切分规则：1.先按空行（段落） 2.再按换行 3.最后按中文应标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节必须带父标题/序号等元信息
        """
        content = section.get("content","") # 从当前章节里拿正文，没有就用空字符串
        if len(content) <= self.config.max_content_length:
            return [section]  # 长度未超限，无需切分，直接返回原章节（列表格式保持统一）
        title = section.get("title","") ## 拿出章节标题，后面切成小块时继续带上
        prefix = f"{title}\n\n" if title else "" #有标题就把标题加在前面，再空两行；没有标题就不加
        available_len = self.config.max_content_length - len(prefix) #计算正文可用长度：总长度-标题前缀长度
        if available_len <= 0: #标题太长，已经没有空间放正文了，就警告并返回空列表
            self.logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
            return []
        body = content #先把章节完整内容放到body里
        if title and body.lstrip().startswith(title): #正文的左边空格去掉并且以title开头
            body = body[body.find(title)+len(title):].lstrip() #找到标题结束的位置，把标题去掉，只留下后面的正文
        #初始化LangChain递归分割器（核心工具：按优先级分隔符切分，保留语义）
        #separators:分隔符优先等级（从粗到细），优先按大语义单元切分，最后才硬拆
        #创建一个“文本切分器对象”，后面拿它去切正文。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = available_len, #正文部分最大长度，之前available_len = self.config.max_content_length - len(prefix)算出来的
            chunk_overlap = 0, #无重叠：按标题切分后语义完整，无需重叠
            separators=[ "\n\n","\n","。","！","？","；",".","!","?",";"," "] #就是一些分隔符等级，先是空格，后换行，然后是中文标点，后英文标点
        )
        sub_sections = [] ## 用来装切得更细的小章节 更细的章节就放在sub_sections这个列表里
        for idx,chunk in enumerate(splitter.split_text(body),start=1): #xxx.split_text(body)是langchain自带的切分工具，enumerate是用来编号的# 先把正文切成小块，再从1开始给每个小块编号
            text = chunk.strip() #文本去掉空格后是text，.strip() 不只是去普通空格，还会去掉两头的换行、Tab 等空白字符。
            if not text: #文本没有就不管直接continue
                continue
            full_text = (prefix + text).strip()  #完整文本full_text就是前缀prefix+纯净文本text，然后这俩去掉空格，把标题和正文拼起来，再去掉最前面和最后面多余的空白
            #现在有了子章节和标题啥的，可以加入sub_sections
            sub_sections.append(
                {
                    "title":f"{title}-{idx}" if title else f"chunk-{idx}", #标题有就是标题和它对应的索引，或者没有就是块对应的索引
                    "content":full_text, #切分后的完整文本
                    "parent_title":title, #父章节标题
                    "part":idx, #就是chunk序号
                    "file_title": section.get("file_title")#所属文件标题
                }
            )
        self.logger.debug(f"超长章节切分完成：{title}->生成{len(sub_sections)}个子chunk")
        return sub_sections
    def _merge_short_section (self,sections:List[Dict[str,str]]) -> List[Dict[str,str]]: #刚那个是太长要切，这个是太短要合并
        """
        最核心注释：
        sections只是原来的小块列表，merged_sections才是去掉已合并重复块后的最终结果
        【辅助函数】过短章节合并（减少碎片化，提升检索效果）
        核心规则：仅仅合并【同父标题】且【当前块长度不足阈值】的相邻chunk，避免跨章节合并
        :param sections: 待合并的chunk列表（通常是_split_long_section切分后的结果）
        :return:合并后的chunk列表，长度适中，保留元信息
        """
        if not sections: #如果参数进来不是sections直接debug返回空列表
            self.logger.debug("待合并Chunk列表为空，直接返回")
            return []
        merged_sections = []  #这里设定一个最终合并列表merged_sections
        current_chunk = None #设定一个现在的块 current_chunk 作为一个暂时存储的东西，
        for sec in sections: #遍历sections
            if current_chunk is None:
                current_chunk = sec
                continue
            # 设定一个bool是is_current_short看看每个sec的content是不是少于config设定的最小值
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            #current_chunk有父标题就拿父标题，没有就拿自己的标题
            current_parent = current_chunk.get("parent_title") or current_chunk.get("title")
            #sec也一样。有父标题就拿父标题，没有就拿自己的标题
            sec_parent = sec.get("parent_title") or sec.get("title")
            #看current_chunk和sec拿到的标题是不是一样
            is_same_parent = current_parent==sec_parent
            #这两个合起来就是合并条件
            if is_current_short and is_same_parent:
                parent_title = sec.get("parent_title","") #先获得父标题parent_title,没有就是空
                next_content = sec["content"]  #这里拿的是sec的content所以是next_content,但是current是上一个所以这里是next_content
                if parent_title and next_content.startswith(parent_title): #如果父目录存在并且下一块文本是父目录开头的
                    next_content = next_content[len(parent_title):].lstrip() #给下一个文本做一下去标题和去左边的空格处理
                current_chunk["content"] += "\n\n" + next_content #当前块太短，就把下一块正文接到后面，但是加了两个换行
                if "part" in sec: #part就是之前细切片出来的序号，这里也要合并
                    current_chunk["part"] = sec["part"] #相当于此时的序号合并到下一个
                self.logger.debug(f"合并短chunk: {current_chunk.get('parent_title')}->累计长度{len(current_chunk['content'])}") # 打印一条调试日志，记录这次合并后的长度
            else:
                merged_sections.append(current_chunk) #不满足合并条件的话，切换为新的待合并块
                current_chunk = sec
                """
                先进入循环，第一次把第一个sec放进`current_chunk`，
                后面每轮sec都自动变成下一个块，然后拿新的`current_chunk`和新的sec比较，
                满足条件就把sec合并进入`current_chunk`  ，不满足条件就把`current_chunk`存起来，
                再让`current_chunk`=sec，这里`current_chunk`就变成了下一个了，继续和下一个sec比较
                """
        if current_chunk is not None: #循环结束后，将最后一个待合并块放进merged_sections
            merged_sections.append(current_chunk)
        self.logger.debug(f"短Chunk合并完成：原{len(sections)}个--->合并后{len(merged_sections)}个")
        return merged_sections
    def _step_5_print_stats(self,line_count:int,sections:List[Dict[str,str]]) -> None:
        """
        核心：把最后切成了多少块、原文多少行这些信息打印出来
        【步骤5】输出文档切分统计信息（日志信息，便于监控/调试）
        :param line_count: MD原始文本总行数
        :param sections: 最终处理后的chunk列表
        :return: 返回sections
        """
        chunk_num = len(sections) #这里是原sections的原始行数chunk_num
        #下面就是搞日志输出：
        self.logger.info("-"*50 + "文档切分统计信息" + "-"*50) #做个分割线
        self.logger.info(f"MD原始文本总函数：{line_count}")
        self.logger.info(f"最终生成的Chunk数量：{chunk_num}")
    def _step_6_backup(self,state:ImportGraphState,sections:List[Dict[str,str]]) -> None:
        """
        【步骤6】把最后切好的chunk保存成一个json文件，留个备份
        :param state: 从这里拿保存位置
        :param sections: 最后切好、合并好的所有chunk
        :return: 无，啥都不返回
        """
        try:
            backup_path = Path("F:/00ZhangGuiZhiKu/backup/")/"data"/state.get("file_title","unknown")/"chunk.json" #自动备份一个文件
            backup_path.parent.mkdir(parents=True, exist_ok=True) #如果文件不存在就创建一个
            with open(backup_path,"w",encoding="utf-8") as f: #打开上面创建的备份路径back_up，然后把备份的存进去
                json.dump(
                    sections,
                    f,
                    ensure_ascii=False,#中文直接保存，不变成\uxxxx
                    indent=2,#自动排版缩进，打开文件时更好看
                )
            #备份成功后打印日志
            self.logger.info(
                f"步骤6:chunk结果备份成功，备份文件路径：{backup_path}"
            )
        except Exception as e:
            self.logger.error(
                f"步骤6：Chunk结果备份失败，错误信息{str(e)}",
                exc_info=False
            )









