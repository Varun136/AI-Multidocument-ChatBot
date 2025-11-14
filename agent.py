import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from config import chat_model_config
load_dotenv(override=True)
from state import AgentState
from vector_db import get_relavent_text, format_vector_data, index_data, extract_image_info


class DocumentEvaluationAgent:
    """Agent to evaluate documents and answer questions based on their content."""

    def __init__(self, collection):
        self._state: AgentState = {
            "messages": [],
            "collection": collection
        }
        self._collection_name = collection
        self._llm = ChatOpenAI(
            model=chat_model_config.get("model_name"),
            base_url=os.environ.get("BASE_URL")
        )

        self._app = None
    

    def load_document(self, file_path: str) -> bool:
        documents = None

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            loader = UnstructuredExcelLoader(file_path)
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            image_info = extract_image_info(file_path)
            documents = [Document(page_content=image_info, metadata={"source": file_path})]
        else:
            return "Unsupported file format"

        documents = loader.load() if not documents else documents
        data = format_vector_data(documents)
        if index_data(self._collection_name, data) != len(data):
            return "Unable to upload the File!"
        return f"File '{os.path.basename(file_path)}' uploaded successfully!"


    def _agent_node(self, state: AgentState) -> AgentState:
        system_message = [SystemMessage(
            content=chat_model_config.get("system_message")
        )]
        response = self._llm.bind_tools([get_relavent_text]).invoke(system_message + state["messages"])
        return {"messages": response}


    def initiate_agent(self, file_name: str):
        self.load_document(file_name)

        graph = StateGraph(state_schema=AgentState)
        graph.add_node('agent_node', self._agent_node)
        graph.add_node("tools", ToolNode([get_relavent_text]))
        graph.add_edge(START, 'agent_node')
        graph.add_conditional_edges(
            "agent_node",
            tools_condition,
            {"tools": "tools", "__end__": END}
        )
        graph.add_edge("tools", "agent_node")
        self._app = graph.compile()
    

    def invoke(self, question: str) -> str:
        self._state["messages"].append(HumanMessage(content=question))
        final_state = self._app.invoke(self._state)
        ai_message = final_state["messages"][-1].content
        return ai_message
    