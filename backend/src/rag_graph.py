from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    question: str
    answer: str

class MultiModalRAGGraph:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = self._initialize_llm()
        self.graph = self._build_graph()
        
    def _initialize_llm(self):
        """初始化LLM"""
        return ChatDeepSeek(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.1
        )
    
    def _retrieve(self, state: State) -> dict:
        """检索相关文档（多模态）- 修复版本"""
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            question = messages[-1].content
        else:
            question = state.get("question", "")
            
        if not question:
            return {"context": "", "question": ""}
            
        try:
            # 使用兼容版本的检索方法
            docs = []
            if hasattr(self.retriever, 'invoke'):
                # 新版本API
                docs = self.retriever.invoke(question)
            elif hasattr(self.retriever, 'get_relevant_documents'):
                # 旧版本API
                docs = self.retriever.get_relevant_documents(question)
            else:
                # 回退到直接调用
                docs = self.retriever(question)
            
            # 构建上下文
            context_parts = []
            for i, doc in enumerate(docs):
                doc_type = doc.metadata.get('type', 'text')
                if doc_type == 'text':
                    context_parts.append(f"文本片段 {i+1}:\n{doc.page_content}")
                else:
                    # 图像文档，添加描述
                    source = doc.metadata.get('source', '未知文档')
                    page = doc.metadata.get('page', 1)
                    context_parts.append(f"相关图像 {i+1}: 来自 {source} 第 {page} 页")
            
            context = "\n\n".join(context_parts)
            
            # 记录检索结果
            text_docs_count = len([d for d in docs if d.metadata.get('type') == 'text'])
            image_docs_count = len([d for d in docs if d.metadata.get('type') == 'image'])
            logger.info(f"检索到 {text_docs_count} 个文本片段和 {image_docs_count} 个图像片段")
            
            return {"context": context, "question": question}
        except Exception as e:
            logger.error(f"检索出错: {e}")
            return {"context": "", "question": question}
    
    def _generate(self, state: State) -> dict:
        """生成回答"""
        context = state.get("context", "")
        question = state.get("question", "")
        
        if not question:
            return {"answer": "请提出您的问题。"}
        
        # 根据是否有图像内容调整提示词
        if "相关图像" in context:
            system_template = """你是一个多模态文档助手，可以处理文本和图像信息。
            
            上下文信息（包含文本和图像描述）：
            {context}
            
            请根据上下文回答问题。如果涉及图像内容，请结合图像描述进行回答。如果上下文没有相关信息，请说明你不知道，不要编造信息。"""
        else:
            system_template = """你是一个专业的文档助手。请根据以下上下文回答问题。
            
            上下文：
            {context}
            
            请根据上下文信息回答问题。如果上下文没有相关信息，请说明你不知道，不要编造信息。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({"context": context, "question": question})
            return {"answer": answer}
        except Exception as e:
            logger.error(f"生成回答出错: {e}")
            return {"answer": f"生成回答时出错: {str(e)}"}
    
    def _format_response(self, state: State) -> dict:
        """格式化响应，添加到消息历史"""
        answer = state.get("answer", "")
        
        return {"messages": [AIMessage(content=answer)]}
    
    def _build_graph(self):
        """构建LangGraph"""
        workflow = StateGraph(State)
        
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("format_response", self._format_response)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    async def ainvoke(self, question: str, config: RunnableConfig = None):
        """异步调用RAG图"""
        if not question or not question.strip():
            return "请输入有效的问题。"
            
        initial_state = State(
            messages=[HumanMessage(content=question)],
            context="",
            question=question,
            answer=""
        )
        
        try:
            if config:
                result = await self.graph.ainvoke(initial_state, config=config)
            else:
                result = await self.graph.ainvoke(initial_state)
            return result.get("answer", "未获得回答。")
        except Exception as e:
            logger.error(f"执行RAG图出错: {e}")
            return f"处理问题时出错: {str(e)}"
    
    def invoke(self, question: str, config: RunnableConfig = None):
        """同步调用RAG图"""
        if not question or not question.strip():
            return "请输入有效的问题。"
            
        initial_state = State(
            messages=[HumanMessage(content=question)],
            context="",
            question=question,
            answer=""
        )
        
        try:
            if config:
                result = self.graph.invoke(initial_state, config=config)
            else:
                result = self.graph.invoke(initial_state)
            return result.get("answer", "未获得回答。")
        except Exception as e:
            logger.error(f"执行RAG图出错: {e}")
            return f"处理问题时出错: {str(e)}"