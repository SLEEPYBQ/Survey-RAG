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


# 修改简洁回答指令变量
concise_instruction = """Please provide your answer in the following format:

[Your concise answer here. If information is not available, write 'N/A']

Source: [Quote the relevant text from the paper here]

Do not include any additional explanations."""

# ...existing code...
def get_questions():
    return [
        {
            "id": "involved_stakeholder",
            "question": (
                "What are the involved stakeholders (e.g., elderly people, caregivers, technical solution providers) in the study? "
                "Stakeholders must meet one of the following criteria: "
                "1. Participate in experiments or studies; "
                "2. Not participate directly but expressed opinions or perspectives (e.g., via interviews, focus groups); "
                "3. Play a role in shaping the findings or conclusions of the paper. "
                + concise_instruction
            )
        },
        {
            "id": "sample_size",
            "question": (
                "What is the sample size of the study? For example, if 100 people participated and only 90 consented to data collection, the sample size is 90. "
                "For multi-study papers, specify the sample size for each study group. "
                + concise_instruction
            )
        },
        {
            "id": "country",
            "question": (
                "What is the country or region of the participants as explicitly stated in the paper (do not infer from the authors’ affiliations)? "
                + concise_instruction
            )
        },
        {
            "id": "age",
            "question": (
                "What age-related information is provided in the study (e.g., age range, mean, or median age)? "
                + concise_instruction
            )
        },
        {
            "id": "gender",
            "question": (
                "What gender-related information is reported in the study? "
                + concise_instruction
            )
        },
        {
            "id": "demographic_background",
            "question": (
                "What demographic background information is reported? (For example, socioeconomic status, educational level, and living context for elderly people or working context for caregivers; also include any additional details such as language proficiency, professional background, or technology literacy if mentioned.) "
                + concise_instruction
            )
        },
        {
            "id": "cognitive_and_physical_impairment",
            "question": (
                "What cognitive and physical impairments are described among the elderly participants? "
                "If standardized measurement tools were used, report the specific scores and the name of the scale; if qualitative terms (e.g., 'mild', 'severe') were used, report them accordingly. "
                + concise_instruction
            )
        },
        {
            "id": "needs_and_expectations",
            "question": (
                "What are the explicitly stated or inferred needs and expectations of users, primarily elderly people and caregivers? "
                "This includes both directly expressed needs and user preferences accompanied by explanatory comments during interviews or post-trial reflections. "
                + concise_instruction
            )
        },
        {
            "id": "application_context",
            "question": (
                "What is the envisioned application context for the robot as explicitly mentioned in the paper? "
                + concise_instruction
            )
        },
        {
            "id": "process_of_the_care",
            "question": (
                "What information is provided about the duration and stage of the care process? "
                "Specify whether the study involved a first encounter, short-term use, or long-term deployment, and include session duration and frequency if available. "
                + concise_instruction
            )
        },
        {
            "id": "methodology",
            "question": (
                "What research methodology was used in the study (e.g., qualitative interviews, quantitative surveys, randomized controlled trials)? "
                + concise_instruction
            )
        },
        {
            "id": "Care_type",
            "question": (
                "What type of care is the study focused on?"
                + concise_instruction
            )
        },
        {
            "id": "robot_type",
            "question": (
                "What type of robot is used in the study? (If the paper uses terms like 'human-like' or 'animal-like', use those directly; otherwise, provide a short description of the robot’s appearance.) "
                + concise_instruction
            )
        },
        {
            "id": "robot_name",
            "question": (
                "What is the name of the robot used in the study? "
                + concise_instruction
            )
        },
        {
            "id": "design_goal",
            "question": (
                "What design goals were set by the solution provider when designing the robot or its interaction functions? "
                + concise_instruction
            )
        },
        {
            "id": "robot_concern_function",
            "question": (
                "What functionalities of the robot were demonstrated, deployed, or introduced to users during the study? "
                + concise_instruction
            )
        },
        {
            "id": "facilitating_functions",
            "question": (
                "What specific robot functions or features are reported to enhance the user experience (i.e., positive features)? "
                "Please provide brief explanations for why these features are considered beneficial. "
                + concise_instruction
            )
        },
        {
            "id": "inhibitory_functions",
            "question": (
                "What specific robot functions or features are reported to hinder the user experience (i.e., negative features)? "
                "Please provide brief explanations for why these features are considered detrimental. "
                + concise_instruction
            )
        },
        {
            "id": "stakeholder_facilitating_characteristics",
            "question": (
                "What characteristics of the stakeholders are associated with better robot use, acceptance, or trust? "
                "Include brief explanations where available. "
                + concise_instruction
            )
        },
        {
            "id": "stakeholder_inhibitory_characteristics",
            "question": (
                "What characteristics of the stakeholders are associated with reduced robot use, lower acceptance, or lower trust? "
                "Include brief explanations where available. "
                + concise_instruction
            )
        },
        {
            "id": "engagement",
            "question": (
                "What evaluation of user engagement in the robot is reported in the study? "
                "This may include quantitative measurements (e.g., rating scales) or qualitative descriptions (e.g., 'high engagement', 'low acceptance', 'gradual trust development'). "
                + concise_instruction
            )
        },
        {
            "id": "acceptance",
            "question": (
                "What evaluation of user acceptance trust in the robot is reported in the study? "
                "This may include quantitative measurements (e.g., rating scales) or qualitative descriptions (e.g., 'high engagement', 'low acceptance', 'gradual trust development'). "
                + concise_instruction
            )
        },
        {
            "id": "trust",
            "question": (
                "What evaluation of user trust in the robot is reported in the study? "
                "This may include quantitative measurements (e.g., rating scales) or qualitative descriptions (e.g., 'high engagement', 'low acceptance', 'gradual trust development'). "
                + concise_instruction
            )
        },
        {
            "id": "key_findings",
            "question": (
                "What are the key findings of the study, as typically summarized in the conclusion or discussion section? "
                + concise_instruction
            )
        },
        {
            "id": "additional_info",
            "question": (
                "What additional information is provided about the study, such as limitations or other relevant details? "
                + concise_instruction
            )
        },
        {
            "id": "testing_context",
            "question": (
                "What is the testing context of the study? (For example, was the test conducted in a lab, care home, hospital, private residence, or another setting?) "
                + concise_instruction
            )
        }
    ]


def query_document_wrapper(args):
    """用于并行处理的查询文档包装器函数"""
    path, question, api_type, api_version, api_base, api_key, doc_name = args
    try:
        success, result = query_document(path, question, api_type, api_version, api_base, api_key)
        return doc_name, success, result
    except Exception as e:
        return doc_name, False, str(e)

def process_question(question, embedding_paths, args, all_results=None):
    """处理单个问题的查询并保存结果为CSV，使用并行处理"""
    api_type = args.api_type
    
    # 准备用于并行处理的参数
    query_args = []
    for path in embedding_paths:
        doc_name = os.path.basename(path)
        if api_type == 'openai':
            query_args.append((path, question['question'], api_type, "2023-05-15", args.api_base, args.api_key, doc_name))
        elif api_type == 'azure':
            query_args.append((path, question['question'], api_type, args.api_version, args.api_endpoint, args.api_key_azure, doc_name))
    
    # 查询结果
    results = []
    
    # 并行执行查询
    print(f"使用 {args.max_workers} 个工作进程并行查询 '{question['id']}'...")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for doc_name, success, result in tqdm(
            executor.map(query_document_wrapper, query_args), 
            total=len(query_args), 
            desc=f"查询 '{question['id']}'"
        ):
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
