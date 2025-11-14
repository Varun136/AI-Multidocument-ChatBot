from pymilvus import MilvusClient
from typing import Any, Dict, Iterable, Annotated
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from state import AgentState
from dotenv import load_dotenv
from config import embedding_model_config, image_extraction_model
import os
import base64
import logging
load_dotenv()


logger = logging.getLogger()
logger.setLevel(logging.INFO)


VECTOR_SIZE = 3072 # text-embedding-3-large vector size.
MILVUS_DB_NAME = "rag_chatbot_qb.db"
MIN_HIT_DISTANCE = 0.2

embedding_client = OpenAIEmbeddings(
    model=embedding_model_config.get("model_name"),
    base_url=os.environ.get("BASE_URL")
)
image_data_extraction_model = ChatOpenAI(
    model=image_extraction_model.get("model_name"),
    base_url=os.environ.get("BASE_URL")
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
milvus_client = MilvusClient(MILVUS_DB_NAME)


def index_data(collection_name: str, data: Dict[str, Any]):
    if not milvus_client.has_collection(collection_name=collection_name):
        logger.info(f"Created collection: {collection_name}")
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=VECTOR_SIZE,
            id_type="int",
        )
    
    try:
        res = milvus_client.insert(collection_name=collection_name, data=data)
    except Exception as e:
        logger.error("Error indexing the data, error: ", str(e))
        return None
    
    logger.info("Data indexed")
    return res.get('insert_count')


def format_vector_data(docs: Iterable[Document], source: str = "NA"):
    documents = text_splitter.split_documents(docs)
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_client.embed_documents(texts)
    return [
        {
            "id": i + 1000,
            "vector": embeddings[i],
            "text": documents[i].page_content,
            "source": documents[i].metadata.get("source", source)
        } for i in range(len(documents))
    ]


# Image embedding utils.
def encode_image(image_path: str):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_image_info(image_path: str):
    base64_image = encode_image(image_path)
    response = image_data_extraction_model.invoke([
        HumanMessage([
            {"type": "text", "text": "Analyze this image in detail and provide the information in the image or about the image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ])
    ])
    logger.info(f"Extracted image info, filename: {image_path}")
    return response.content


# Relavent tool for the system.
@tool(name_or_callable="get_relavent_text")
def get_relavent_text(query: str, state: Annotated[AgentState, InjectedState]):
    """Retrieve relavent document from the vector database
    Args:
        query: The query that needs to be searched in the vector database
    """
    logger.info(f"Tool invoked 'get_relavent_text' with query {query}")

    collection_name = state["collection"]
    query_vectors = embedding_client.embed_documents([query])  
    try:    
        res = milvus_client.search(
            collection_name=collection_name, 
            data=query_vectors,
            limit=3,
            output_fields=["text"],
            )
    except Exception as e:
        return f"Docs were not properly indexed, error: {e}"
        
    relavent_docs = []
    for hit in res[0]:
        if hit.get("distance") > MIN_HIT_DISTANCE:
            relavent_docs.append(hit.get("entity").get("text"))
    
    relavent_response = f"Relavent docs: \n {'\n\n'.join(relavent_docs)}"
    return relavent_response

