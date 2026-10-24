import json
import logging
from typing import List,Dict
from processor.import_processor.base import BaseNode,setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings
class NodeBGEEmbedding(BaseNode): #用BGE-M3把所有文本向量化然后存进去
    #这个节点就干一件事就是把chunks更新加入稠密向量和稀疏向量这样
    name:str = "node_bge_embedding"
    def process(self,state:ImportGraphState) -> ImportGraphState:
        chunks = self._step_1_validate_input(state) #第一个方法：校验下数据，得到chunks切片
        output_data = self._step_2_generate_embedding(chunks) #第二个方法：输入chunks切片，得到稠密向量和稀疏向量
        state['chunks'] = output_data #第三个方法：更新全局state状态，将带向量的chunks回传下游
        return state
    def _step_1_validate_input(self,state:ImportGraphState) -> List[Dict]:
        chunks = state.get('chunks')
        if not chunks: #判断切片是否存在
            raise StateFieldError(
            field_name="chunks",
            message="chunks不能为空",
            expected_type=list)
        if not isinstance(chunks, list): #判断chunks是不是list列表形式
            raise StateFieldError(
                field_name="chunks",
                message="chunks数据类型不正确",
                expected_type=list)
        return chunks
    def _step_2_generate_embedding(self,chunks: List[Dict[str,str]]) -> List[Dict]: #输入chunks生成两个向量把他俩加进去就行了
        output_data = [] #创建一个output_data输出结果列表
        batch_size = 5 #每次只拿5个一起处理
        for i in range(0,len(chunks),batch_size): #从0开始到chunks长度遍历，每次i+5，举例子：len(chunks) = 12，batch_size = 5，那么0，5，10
            batch_texts = chunks[i:i+batch_size] #用batch_texts去接这个切出来的chunks，然后这个batch_texts也是个列表
            input_texts = [] #这里建一个列表用来存放一次次循环出来的前5个chunks切片
            for doc in batch_texts: #遍历这个取出来的只有前5个chunks的列表
                item_name = doc["item_name"]
                content = doc["content"]
                input_texts.append(
                    f"{item_name}\n{content}"
                    if item_name else content) #这里会得到input_texts列表，列表举例：这是三元表达式，如果if走前面的式子，不行就走else
                # input_texts = ["华为Mate80\n长按电源键开机","华为Mate80\n连接充电器充电","华为Mate80\n进入设置恢复出厂设置]
            docs_embedding = generate_embeddings(input_texts) #得到稠密向量和稀疏向量一个大字典，doc_embedding是得到的向量切片
            for j,doc in enumerate(batch_texts):
                item = doc.copy() #先要复制每次取到的前五个chunks切片中的一个复制为item
                item["dense_vector"] = (docs_embedding["dense"][j]) #从大字典里往外拿稠密向量
                item["sparse_vector"] = (docs_embedding["sparse"][j]) #从大字典里往外拿稀疏向量
                output_data.append(item) #output_data这个列表加入进有向量的item
            self.logger.info(
                f"成功获取第"
                f"{i+1}-"
                f"{min(i+len(batch_texts),len(chunks))}"
                f"项的嵌入"
            )
        return output_data
