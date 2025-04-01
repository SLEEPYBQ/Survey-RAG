import argparse
import os
import time
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import textract
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

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
    pdf_path, output_folder, api_base, api_key = args
    
    # 设置API凭证 (在每个进程中都需要设置)
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_API_KEY"] = api_key
    
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
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents([document], embeddings)
        
        # 生成输出文件名（去掉.pdf后缀）
        output_name = pdf_file.replace('.pdf', '')
        output_path = os.path.join(output_folder, output_name)
        
        # 保存向量存储
        db.save_local(output_path)
        
        return pdf_file, True, output_path
    except Exception as e:
        return pdf_file, False, str(e)
    

def query_document(embedding_path, query, api_base, api_key):
    """使用LLM查询文档内容"""
    # 设置API凭证
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        # 加载embedding
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(embedding_path, embeddings)
        
        # 创建LLM模型和检索链
        llm_model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm_model, db.as_retriever())
        
        # 执行查询
        result = qa({"question": query, "chat_history": []})
        return True, result['answer']
    except Exception as e:
        return False, str(e)

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='并行处理PDF文件并提取信息')
    
    # 添加参数 - 保留原有参数
    parser.add_argument('--input_dir', type=str, required=True, help='PDF文件的输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='Embedding的输出目录')
    parser.add_argument('--api_base', type=str, default="https://api.openai-proxy.org/v1", help='OpenAI API基础URL')
    parser.add_argument('--api_key', type=str, default="sk-lTSVS2yip8kbDNpYj5GfwmW5RFJOOrf33zX3gh55xZ2KSWlH", help='OpenAI API密钥')
    parser.add_argument('--max_workers', type=int, default=multiprocessing.cpu_count(), help='最大并行工作进程数')
    
    # 添加查询相关参数
    parser.add_argument('--mode', type=str, choices=['process', 'query', 'both'], default='both', 
                        help='运行模式: process=仅处理PDF, query=仅查询, both=处理并查询')
    parser.add_argument('--query', type=str, default="请总结这篇文献的主要内容", 
                        help='查询文本, 如"请提取文献中的研究方法和结果"')
    parser.add_argument('--save_results', type=bool, default=True, 
                        help='是否将查询结果保存到文件')
    
    # 解析参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理PDF阶段
    if args.mode in ['process', 'both']:
        # 收集所有PDF文件
        pdf_files = []
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            print(f"在 {args.input_dir} 中未找到PDF文件")
            return
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 准备并行处理参数
        process_args = [(pdf_path, args.output_dir, args.api_base, args.api_key) for pdf_path in pdf_files]
        
        # 并行处理PDF文件
        start_time = time.time()
        results = []
        
        print(f"使用 {args.max_workers} 个工作进程进行并行处理...")
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for result in tqdm(executor.map(process_single_pdf, process_args), total=len(pdf_files), desc="处理PDF"):
                results.append(result)
        
        end_time = time.time()
        
        # 输出结果
        success_count = sum(1 for _, success, _ in results if success)
        
        print("\nPDF处理完成!")
        print(f"总共处理: {len(pdf_files)} 个文件")
        print(f"成功处理: {success_count} 个文件")
        print(f"失败处理: {len(pdf_files) - success_count} 个文件")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        
        # 输出失败的文件
        if len(pdf_files) - success_count > 0:
            print("\n失败的文件:")
            for file, success, error in results:
                if not success:
                    print(f" - {file}: {error}")
        
        # 创建一个简单的索引文件，列出所有成功处理的PDF
        index_path = os.path.join(args.output_dir, "index.txt")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"处理日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"提示词: {args.query}\n\n")
            f.write("成功处理的文件:\n")
            for file, success, path in results:
                if success:
                    f.write(f"{file}: {path}\n")
        
        print(f"\n索引文件已保存至: {index_path}")
    
    # 查询阶段
    if args.mode in ['query', 'both']:
        print("\n开始查询文档...")
        
        # 获取所有embedding文件夹
        embedding_paths = []
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isdir(item_path) and 'index.faiss' in os.listdir(item_path):
                embedding_paths.append(item_path)
        
        if not embedding_paths:
            print(f"在 {args.output_dir} 中未找到embedding文件")
            return
        
        print(f"找到 {len(embedding_paths)} 个embedding文件")
        
        # 查询结果
        query_results = {}
        
        # 对每个文档进行查询
        for path in tqdm(embedding_paths, desc="查询文档"):
            doc_name = os.path.basename(path)
            success, result = query_document(path, args.query, args.api_base, args.api_key)
            query_results[doc_name] = {
                "success": success,
                "result": result
            }
        
        # 打印和保存结果
        print("\n查询结果:")
        for doc, res in query_results.items():
            print(f"\n文档: {doc}")
            if res["success"]:
                print(f"结果: {res['result']}")
            else:
                print(f"查询失败: {res['result']}")
        
        # 保存结果到文件
        if args.save_results:
            results_path = os.path.join(args.output_dir, "query_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": args.query,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "results": query_results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n查询结果已保存至: {results_path}")

if __name__ == "__main__":
    main()