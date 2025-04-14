import textract
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 保留原有的PDF处理函数
def extract_text_from_pdf(pdf_path):
    """从PDF提取文本内容"""
    try:
        doc = textract.process(pdf_path)
        return doc.decode('utf-8')
    except Exception as e:
        print(f"处理PDF时出错: {pdf_path}, 错误信息: {e}")
        return ""

def process_single_pdf(args):
    """处理单个PDF文件并保存embedding - 使用文档分块策略"""
    pdf_path, output_folder, api_type, api_base, api_key, api_version, api_endpoint, api_key_azure = args

    if api_type == 'openai':
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_KEY"] = api_key
    elif api_type == 'azure':
        os.environ["OPENAI_API_VERSION"] = api_version
        os.environ["AZURE_OPENAI_ENDPOINT"] = api_endpoint
        os.environ["AZURE_OPENAI_API_KEY"] = api_key_azure
    
    # 获取文件名
    pdf_file = os.path.basename(pdf_path)
    
    # 提取文本
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        return pdf_file, False, "无法提取文本"
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # 每块大约1000字符
            chunk_overlap=200,      # 相邻块之间重叠200字符，提高连贯性
            separators=["\n\n", "\n", ". ", " ", ""],  # 按自然段落、句子等进行分割
        )
        
        # 分割文本成块
        chunks = text_splitter.split_text(text)
        
        # 为每个块创建Document对象
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_file, 
                        "path": pdf_path,
                        "chunk": i,
                        "total_chunks": len(chunks)
                    }
                )
            )
        
        # 创建向量存储
        if api_type == 'openai':
            embeddings = OpenAIEmbeddings()
        elif api_type == 'azure':
            embeddings = AzureOpenAIEmbeddings()
        
        db = FAISS.from_documents(documents, embeddings)
        
        # 生成输出文件名（去掉.pdf后缀）
        output_name = pdf_file.replace('.pdf', '')
        output_path = os.path.join(output_folder, output_name)
        
        # 保存向量存储
        db.save_local(output_path)
        
        return pdf_file, True, output_path
    except Exception as e:
        return pdf_file, False, str(e)