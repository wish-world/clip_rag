from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    question: str
    answer: str

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.1
)

class RAGGraph:
    def __init__(self, retriever):
        self.retriever = retriever
        self.graph = self._build_graph()
        
    def _retrieve(self, state: State) -> dict:
        """检索相关文档"""
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            question = messages[-1].content
        else:
            question = state.get("question", "")
            
        if not question:
            return {"context": "", "question": ""}
            
        try:
            docs = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            return {"context": context, "question": question}
        except Exception as e:
            print(f"检索出错: {e}")
            return {"context": "", "question": question}
    
    def _generate(self, state: State) -> dict:
        """生成回答"""
        context = state.get("context", "")
        question = state.get("question", "")
        
        if not question:
            return {"answer": "请提出您的问题。"}
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的文档助手。请根据以下上下文回答问题。
            
            上下文：
            {context}
            
            请根据上下文信息回答问题。如果上下文没有相关信息，请说明你不知道，不要编造信息。
            回答要简洁明了，重点突出。"""),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            answer = chain.invoke({"context": context, "question": question})
            return {"answer": answer}
        except Exception as e:
            print(f"生成回答出错: {e}")
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
    
    async def ainvoke(self, question: str):
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
            result = await self.graph.ainvoke(initial_state)
            return result.get("answer", "未获得回答。")
        except Exception as e:
            print(f"执行RAG图出错: {e}")
            return f"处理问题时出错: {str(e)}"