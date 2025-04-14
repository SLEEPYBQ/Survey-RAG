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
    """使用LLM查询文档内容 - 使用改进的检索策略"""
    try:
        # 加载embedding
        if api_type == 'openai':
            embeddings = OpenAIEmbeddings(
                api_key=api_key,
                base_url=api_base
            )
        elif api_type == 'azure':
            embeddings = AzureOpenAIEmbeddings(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )
        
        db = FAISS.load_local(embedding_path, embeddings)
        

        retriever = db.as_retriever(
            search_type="mmr",          
            search_kwargs={
                "k": 60,                 
                "fetch_k": 90,          
                "lambda_mult": 0.85      
            }
        )
        
        # 创建LLM模型
        if api_type == 'openai':
            llm_model = ChatOpenAI(
                temperature=0.3, 
                model="gpt-4o-mini",
                api_key=api_key,
                base_url=api_base
            )
        elif api_type == 'azure':
            llm_model = AzureChatOpenAI(
                temperature=0.3, 
                model="gpt-4o-mini",
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )
            
        qa = ConversationalRetrievalChain.from_llm(llm_model, retriever)
        
        # 执行查询
        # result = qa({"question": query, "chat_history": []})
        result = qa.invoke({"question": query, "chat_history": []})
        # print(f"查询结果: {result.get('answer')}")
        return True, result['answer']
    except Exception as e:
        return False, str(e)