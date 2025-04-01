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

from embedding import extract_text_from_pdf, process_single_pdf
from retrieval import query_document
from utils import get_questions, process_question, save_consolidated_csv


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='并行处理PDF文件并提取指定信息')
    
    # 添加基本参数
    parser.add_argument('--input_dir', type=str, required=True, help='PDF文件的输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='Embedding和结果的输出目录')
    # --api_base和--api_key设置自己的, 我使用的是CloseAI的中转API
    parser.add_argument('--api_base', type=str, default="https://api.openai-proxy.org/v1", help='OpenAI API基础URL')
    parser.add_argument('--api_key', type=str, default="<YOUR-API-KEY>", help='OpenAI API密钥')
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
            
            # 生成合并CSV文件
            if args.consolidated_csv and all_results:
                save_consolidated_csv(all_results, questions, args.output_dir)

if __name__ == "__main__":
    main()