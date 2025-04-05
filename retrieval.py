import argparse
import os
import time
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import textract
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def query_document(embedding_path, query, api_type, api_version, api_base, api_key):
    """使用LLM查询文档内容"""
    # 设置API凭证
    if api_type == 'openai':
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_KEY"] = api_key
    elif api_type == 'azure':
        os.environ["OPENAI_API_VERSION"] = api_version
        os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
        # os.environ["OPENAI_API_KEY"] = api_key

    
    try:
        # 加载embedding
        if api_type == 'openai':
            embeddings = OpenAIEmbeddings()
        elif api_type == 'azure':
            embeddings = AzureOpenAIEmbeddings()
        db = FAISS.load_local(embedding_path, embeddings)
        
        # 创建LLM模型和检索链
        if api_type == 'openai':
            llm_model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
        elif api_type == 'azure':
            llm_model = AzureChatOpenAI(temperature=0.3, model="gpt-4o-mini")
        # llm_model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
        qa = ConversationalRetrievalChain.from_llm(llm_model, db.as_retriever())
        
        # 执行查询
        result = qa({"question": query, "chat_history": []})
        return True, result['answer']
    except Exception as e:
        return False, str(e)