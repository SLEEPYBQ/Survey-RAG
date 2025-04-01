import textract
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# 设置API凭证
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-lTSVS2yip8kbDNpYj5GfwmW5RFJOOrf33zX3gh55xZ2KSWlH"

def extract_text_from_pdf(pdf_path):
    """从PDF提取文本"""
    try:
        doc = textract.process(pdf_path)
        return doc.decode('utf-8')
    except Exception as e:
        print(f"处理PDF时出错: {e}")
        return ""

def create_and_save_embeddings(pdf_folder, output_folder):
    """为每个PDF创建embedding并保存"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 初始化嵌入模型
    embeddings = OpenAIEmbeddings()
    
    # 处理每个PDF
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"处理: {pdf_file}")
        
        # 从PDF提取文本
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"无法从 {pdf_file} 提取文本，跳过")
            continue
        
        # 创建Document对象
        document = Document(
            page_content=text,
            metadata={"source": pdf_file}
        )
        
        # 创建向量存储
        db = FAISS.from_documents([document], embeddings)
        
        # 生成输出文件名（去掉.pdf后缀）
        output_name = pdf_file.replace('.pdf', '')
        output_path = os.path.join(output_folder, output_name)
        
        # 保存向量存储
        db.save_local(output_path)
        print(f"已保存embedding到: {output_path}")
    
    return pdf_files

def load_specific_embedding(embedding_folder, pdf_name):
    """加载特定PDF的embedding"""
    # 生成文件名（去掉.pdf后缀）
    if pdf_name.endswith('.pdf'):
        pdf_name = pdf_name.replace('.pdf', '')
    
    embedding_path = os.path.join(embedding_folder, pdf_name)
    
    # 加载嵌入模型
    embeddings = OpenAIEmbeddings()
    
    # 加载向量存储
    db = FAISS.load_local(embedding_path, embeddings)
    return db

def main():
    pdf_folder = "./pdf"
    embedding_folder = "./embeddings"
    
    # 检查是否已经创建了embeddings
    if not os.path.exists(embedding_folder) or len(os.listdir(embedding_folder)) == 0:
        print("首次运行，创建所有PDF的embeddings...")
        pdf_files = create_and_save_embeddings(pdf_folder, embedding_folder)
    else:
        pdf_files = [f.replace('.pdf', '') for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        print("已检测到embeddings文件夹，跳过创建步骤")
    
    # 打印可用的PDF文件
    print("\n可用的PDF文件:")
    for i, pdf in enumerate(pdf_files):
        print(f"{i+1}. {pdf}")
    
    # 设置聊天模型
    llm_model = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
    
    while True:
        # 用户选择特定PDF
        try:
            choice = input("\n选择PDF文件编号 (或输入 'exit' 退出): ")
            
            if choice.lower() == 'exit':
                break
            
            idx = int(choice) - 1
            if idx < 0 or idx >= len(pdf_files):
                print("无效的选择，请重试")
                continue
            
            selected_pdf = pdf_files[idx]
            print(f"已选择: {selected_pdf}")
            
            # 加载特定PDF的embedding
            db = load_specific_embedding(embedding_folder, selected_pdf)
            
            # 创建问答链
            qa = ConversationalRetrievalChain.from_llm(llm_model, db.as_retriever())
            chat_history = []
            
            # 进入该PDF的问答模式
            print(f"\n正在与 {selected_pdf} 交流中... (输入 'back' 返回PDF选择)")
            while True:
                query = input("\n输入问题: ")
                
                if query.lower() == 'back':
                    break
                    
                result = qa({"question": query, "chat_history": chat_history})
                chat_history.append((query, result['answer']))
                print(f"\n回答: {result['answer']}")
            
        except ValueError:
            print("请输入有效的数字")
        except Exception as e:
            print(f"发生错误: {e}")
    
    print("程序已退出")

if __name__ == "__main__":
    main()