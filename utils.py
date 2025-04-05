import argparse
import os
import time
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import textract
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from retrieval import query_document

# 定义简洁回答指令变量
concise_instruction = "Please provide a concise answer without additional explanations. If the information is not available, simply respond with 'N/A'. "

# 定义所有问题列表
def get_questions():
    return [
        {"id": "stakeholder", "question": concise_instruction + "What are the involved stakeholders? For example: older adults, caregivers, domain experts, solution providers, etc."},
        {"id": "sample_size", "question": concise_instruction + "What is the sample size of the study?"},
        {"id": "country", "question": concise_instruction + "In which country or countries were the participants located?"},
        {"id": "demographic", "question": concise_instruction + "What are the demographics of the participants in the study?"},
        {"id": "impairment", "question": concise_instruction + "What cognitive or physical impairments do the participants have, if any?"},
        {"id": "needs", "question": concise_instruction + "What needs are addressed by the robot? Or what are the needs of the stakeholders? For example: chatting, reminding, daily routine assistance, exercise guidance, etc."},
        {"id": "context", "question": concise_instruction + "What is the context of the study? For example, what type of community, level of care facility, or environment are the older adults in?"},
        {"id": "care_process", "question": concise_instruction + "What is the process of care described in the study? Is it early stage or long-term care? Is it first contact?"},
        {"id": "methodology", "question": concise_instruction + "What methodology was used in this paper? For example: focus group, one-to-one interview, questionnaire, experiment, etc."},
        {"id": "care_type", "question": concise_instruction + "What type of care is provided or discussed in the study?"},
        {"id": "robot_type", "question": concise_instruction + "Which type of robot is used in the study? For example: humanoid, machine-like, animal-like, etc."},
        {"id": "robot_name", "question": concise_instruction + "What is the name of the robot used in the study?"},
        {"id": "robot_function", "question": concise_instruction + "What is the general function of the robot in the study?"},
        {"id": "facilitating_functions", "question": concise_instruction + "What specific functions of the robot facilitate care or assistance?"},
        {"id": "inhibitory_functions", "question": concise_instruction + "What specific functions of the robot inhibit or hinder care or assistance?"},
        {"id": "stakeholder_facilitating", "question": concise_instruction + "What characteristics of the stakeholders facilitate the use of the robot?"},
        {"id": "stakeholder_inhibitory", "question": concise_instruction + "What characteristics of the stakeholders inhibit or hinder the use of the robot?"},
        {"id": "engagement", "question": "How is user engagement with the robot described or measured in the study? If applicable, include the measurement results (e.g., whether engagement increased or decreased). If not, response should be 'N/A'."},
        {"id": "acceptance", "question": "How is user acceptance of the robot described or measured in the study? If applicable, include the measurement results (e.g., whether acceptance increased or decreased). If not, response should be 'N/A'."},
        {"id": "trust", "question": "How is trust in the robot addressed or measured in the study? If applicable, include the measurement results (e.g., whether trust increased or decreased). If not, response should be 'N/A'."},
        {"id": "key_findings", "question": concise_instruction + "What are the key findings of the study?"},
        {"id": "additional_info", "question": concise_instruction + "What additional information is relevant from this study that doesn't fit into the categories above?"}
    ]

def process_question(question, embedding_paths, args, all_results=None):
    """处理单个问题的查询并保存结果为CSV"""
    # 查询结果
    results = []

    api_type = args.api_type
    
    # 对每个文档进行查询
    for path in tqdm(embedding_paths, desc=f"查询 '{question['id']}'"):
        doc_name = os.path.basename(path)
        if api_type == 'openai':
            success, result = query_document(path, question['question'], api_type, "2023-05-15", args.api_base, args.api_key)
        elif api_type == 'azure':
            success, result = query_document(path, question['question'], api_type, args.api_version, args.api_endpoint, args.api_key_azure)

        
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
