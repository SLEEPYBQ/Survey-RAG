import argparse
import os
import time
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import textract
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

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

# 定义所有问题列表
def get_questions():
    return [
        {"id": "stakeholder", "question": "Involved stakeholder? 参与的利益相关者？"},
        {"id": "sample_size", "question": "Sample size 样本大小"},
        {"id": "country", "question": "Participant country 参与者所在国家"},
        {"id": "demographic", "question": "Participant demographic 参与者人口统计信息"},
        {"id": "impairment", "question": "Cognitive and physical impairment 认知和身体障碍"},
        {"id": "needs", "question": "Needs and expectations? 需求和期望？"},
        {"id": "context", "question": "Context: 什么样的社区，什么级别的护理机构，老年人所在的社区或环境"},
        {"id": "care_process", "question": "Process of the care: 前期还是长期；第一次接触？"},
        {"id": "methodology", "question": "Methodology 方法论"},
        {"id": "care_type", "question": "Care type 护理类型"},
        {"id": "robot_type", "question": "Robot type (embodiment) 机器人类型（具象化）"},
        {"id": "robot_name", "question": "Robot name 机器人名称"},
        {"id": "robot_function", "question": "Robot general function 机器人一般功能"},
        {"id": "facilitating_functions", "question": "Facilitating functions (specific) 促进功能（具体）"},
        {"id": "inhibitory_functions", "question": "Inhibitory functions (specific) 抑制功能（具体）"},
        {"id": "stakeholder_facilitating", "question": "Stakeholder facilitating characteristics 利益相关者促进特征"},
        {"id": "stakeholder_inhibitory", "question": "Stakeholder inhibitory characteristics 利益相关者抑制特征"},
        {"id": "engagement", "question": "Engagement 参与度"},
        {"id": "acceptance", "question": "Acceptance 接受度"},
        {"id": "trust", "question": "Trust 信任"},
        {"id": "key_findings", "question": "Key findings (brief) 关键发现（简要）"},
        {"id": "additional_info", "question": "Additional info 其他信息"}
    ]

def process_question(question, embedding_paths, args, all_results=None):
    """处理单个问题的查询并保存结果为CSV"""
    # 查询结果
    results = []
    
    # 对每个文档进行查询
    for path in tqdm(embedding_paths, desc=f"查询 '{question['id']}'"):
        doc_name = os.path.basename(path)
        success, result = query_document(path, question['question'], args.api_base, args.api_key)
        
        if success:
            result_item = {
                "document": doc_name,
                "question_id": question['id'],
                "question": question['question'],
                "result": result
            }
            results.append(result_item)
            
            # 如果提供了all_results字典，将结果也存到那里
            if all_results is not None:
                if doc_name not in all_results:
                    all_results[doc_name] = {"document": doc_name}
                all_results[doc_name][question['id']] = result
        else:
            print(f"查询 '{doc_name}' 失败: {result}")
            # 如果提供了all_results字典，添加错误信息
            if all_results is not None:
                if doc_name not in all_results:
                    all_results[doc_name] = {"document": doc_name}
                all_results[doc_name][question['id']] = f"错误: {result}"
    
    # 保存结果到CSV
    csv_path = os.path.join(args.output_dir, f"results_{question['id']}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['document', 'question_id', 'question', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"已保存 '{question['id']}' 的查询结果到: {csv_path}")
    
    return results

def save_consolidated_csv(all_results, questions, output_dir):
    """将所有问题的结果保存到一个合并的CSV文件中"""
    # 构建CSV的字段名（表头）
    fieldnames = ['document'] + [q['id'] for q in questions]
    
    # 保存路径
    consolidated_csv_path = os.path.join(output_dir, "all_results_consolidated.csv")
    
    # 将结果保存为CSV
    with open(consolidated_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 写入每个文档的所有问题结果
        for doc_name, results in all_results.items():
            writer.writerow(results)
    
    print(f"\n已保存所有结果到合并CSV文件: {consolidated_csv_path}")

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='并行处理PDF文件并提取指定信息')
    
    # 添加基本参数
    parser.add_argument('--input_dir', type=str, required=True, help='PDF文件的输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='Embedding和结果的输出目录')
    parser.add_argument('--api_base', type=str, default="https://api.openai-proxy.org/v1", help='OpenAI API基础URL')
    parser.add_argument('--api_key', type=str, default="sk-lTSVS2yip8kbDNpYj5GfwmW5RFJOOrf33zX3gh55xZ2KSWlH", help='OpenAI API密钥')
    parser.add_argument('--max_workers', type=int, default=multiprocessing.cpu_count(), help='最大并行工作进程数')
    
    # 添加查询相关参数
    parser.add_argument('--mode', type=str, choices=['process', 'query', 'both'], default='both', 
                        help='运行模式: process=仅处理PDF, query=仅查询, both=处理并查询')
    parser.add_argument('--question_id', type=str, 
                        help='要查询的问题ID，例如"stakeholder"。如果未指定，将显示所有可用问题')
    parser.add_argument('--all_questions', action='store_true', 
                        help='查询所有问题（一次一个）并保存为单独的CSV文件')
    parser.add_argument('--consolidated_csv', action='store_true', default=True,
                        help='除单独的CSV外，是否还生成一个包含所有问题的合并CSV文件')
    
    # 解析参数
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有问题列表
    questions = get_questions()
    
    # 如果未指定问题ID但要求查询，显示所有可用问题
    if args.mode in ['query', 'both'] and not args.question_id and not args.all_questions:
        print("\n可用的问题列表:")
        for i, q in enumerate(questions):
            print(f"{i+1}. ID: {q['id']} - {q['question']}")
        print("\n请使用 --question_id 参数指定要查询的问题，或使用 --all_questions 参数查询所有问题")
        return
    
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
    
    # 查询阶段
    if args.mode in ['query', 'both']:
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
        
        # 用于存储所有问题的查询结果，以便生成合并CSV
        all_results = {} if args.consolidated_csv or args.all_questions else None
        
        # 如果指定了查询单个问题
        if args.question_id and not args.all_questions:
            # 查找指定问题
            selected_question = None
            for q in questions:
                if q['id'] == args.question_id:
                    selected_question = q
                    break
            
            if not selected_question:
                print(f"未找到ID为 '{args.question_id}' 的问题")
                return
            
            print(f"\n查询问题: {selected_question['question']}")
            process_question(selected_question, embedding_paths, args, all_results)
            
            # 如果只有一个问题但要求生成合并CSV，也生成一个单列的合并文件
            if args.consolidated_csv and all_results:
                save_consolidated_csv(all_results, [selected_question], args.output_dir)
        
        # 如果要查询所有问题
        elif args.all_questions:
            print("\n开始依次查询所有问题...")
            for i, question in enumerate(questions):
                print(f"\n[{i+1}/{len(questions)}] 查询问题: {question['question']}")
                process_question(question, embedding_paths, args, all_results)
                
                # 避免API限流
                if i < len(questions) - 1:  # 不是最后一个问题
                    print("等待3秒后继续下一个问题...")
                    time.sleep(3)
            
            # 生成合并CSV文件
            if args.consolidated_csv and all_results:
                save_consolidated_csv(all_results, questions, args.output_dir)

if __name__ == "__main__":
    main()