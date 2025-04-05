import textract
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

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
    """处理单个PDF文件并保存embedding"""
    pdf_path, output_folder, api_type, api_base, api_key, api_version, api_endpoint, api_key_azure  = args

    
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
        # 创建Document对象
        document = Document(
            page_content=text,
            metadata={"source": pdf_file, "path": pdf_path}
        )
        
        # 创建向量存储
        if api_type == 'openai':
            embeddings = OpenAIEmbeddings()
        elif api_type == 'azure':
            embeddings = AzureOpenAIEmbeddings()
        # embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents([document], embeddings)
        
        # 生成输出文件名（去掉.pdf后缀）
        output_name = pdf_file.replace('.pdf', '')
        output_path = os.path.join(output_folder, output_name)
        
        # 保存向量存储
        db.save_local(output_path)
        
        return pdf_file, True, output_path
    except Exception as e:
        return pdf_file, False, str(e)