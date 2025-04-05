# Survey-RAG

Survey-RAG 是一个用于处理学术调查PDF文档并使用大语言模型进行信息提取的工具。该工具使用向量数据库和检索增强生成（RAG）来高效地从多个PDF文件中提取结构化信息。

## 功能

- 并行处理多个PDF文件
- 将PDF内容转换为向量嵌入并存储
- 使用预定义问题集自动查询文档内容
- 支持生成单独的问题CSV结果或合并的综合CSV报告

## 依赖库

- `argparse`：命令行参数解析
- `os`：操作系统接口
- `time`：时间相关功能
- `csv`：CSV文件读写
- `multiprocessing`：多进程支持
- `concurrent.futures`：并行执行工具
- `textract`：从PDF提取文本
- `tqdm`：进度条显示
- `langchain_openai`：OpenAI接口
- `langchain_community.vectorstores`：向量数据库
- `langchain.schema`：LangChain文档模式
- `langchain.chains`：检索链

## How it works?

Survey-RAG 主要通过以下几个步骤进行操作：

1. 它首先处理指定文件夹中的 PDF 文档，提取文本并使用 Hugging Face Transformers 库将其拆分成可管理的块。

2. 然后，使用 LangChain 库通过默认的 OpenAI 嵌入模型（text-embedding-ada-002）对每个文本块进行嵌入。

3. 这些嵌入被存储在 FAISS 索引中，提供了一种紧凑高效的存储方式。

4. 最后，查询接口允许从已索引的数据中检索相关信息。该应用程序会提取并显示最相关的文本块。

![Untitled-2023-06-16-1537](https://github.com/raghavan/PdfGptIndexer/assets/131585/2e71dd82-bf4f-44db-b1ae-908cbb465deb)




## 安装

1. 克隆此仓库：

```bash
git clone https://github.com/SLEEPYBQ/Survey-RAG.git
cd Survey-RAG
```

2. 安装依赖库：

```bash
pip install argparse textract tqdm langchain-openai langchain-community faiss-cpu
```

## 使用方法

### 基本用法

```bash
python main.py --input_dir /path/to/pdfs --output_dir /path/to/output --api_key your_openai_api_key
```


### 参数说明

- `--input_dir`：PDF文件的输入目录（必填）
- `--output_dir`：输出结果的目录（必填）
- `--api_base`：OpenAI API基础URL（默认为"https://api.openai-proxy.org/v1"）
- `--api_key`：OpenAI API密钥
- `--max_workers`：最大并行处理工作进程数（默认为CPU核心数）
- `--mode`：运行模式（'process'=仅处理PDF, 'query'=仅查询, 'both'=处理并查询，默认为'both'）
- `--question_id`：要查询的特定问题ID
- `--all_questions`：查询所有预定义问题
- `--consolidated_csv`：生成包含所有问题结果的综合CSV文件（默认开启）
- `--api_type`：OpenAI API类型, 专用于HKUST ITSC Azure（需要选择是"openai"还是"azure"）
- `--api_version`：OpenAI API版本, 专用于HKUST ITSC Azure（默认为"2023-05-15"）
- `--api_endpoint`：OpenAI API端点, 专用于HKUST ITSC Azure（默认为"https://hkust.azure-api.net"）
- `--api_key_azure'`：HKUST ITSC Azure的API Key


### 运行模式示例

#### 区分openai和HKUST ITSC Azure
key可以直接在main.py硬编码, <YOUR-API-KEY>处

1. openai
```bash
python main.py --input_dir ./pdfs --output_dir ./embeddings --mode both --all_questions --api_type openai
```
2. HKUST ITSC Azure

```bash
python main.py --input_dir ./pdfs --output_dir ./embeddings --mode both --all_questions --api_type azure
```

1. 仅处理PDF并生成嵌入：

```bash
python main.py --input_dir /path/to/pdfs --output_dir /path/to/output --api_key your_api_key --mode process
```

2. 仅查询已处理的PDF嵌入（特定问题）：

```bash
python main.py --output_dir /path/to/output --api_key your_api_key --mode query --question_id stakeholder
```

3. 查询所有问题并生成合并报告：

```bash
python main.py --output_dir /path/to/output --api_key your_api_key --mode query --all_questions
```

## 预定义问题

工具包含22个预定义问题，涵盖了学术调查的多个方面，包括：

- 利益相关者信息
- 样本大小和人口统计
- 参与者国家和特征
- 认知和身体障碍
- 需求和期望
- 护理过程和类型
- 机器人类型和功能
- 参与度、接受度和信任度
- 关键发现

## 输出结果

- 每个问题的单独CSV文件：`results_[question_id].csv`
- 合并所有问题的综合报告：`all_results_consolidated.csv`

## Acknowledgements
该项目受到了以下项目的启发,并复用其部分代码： 
1. [PdfGptIndexer](https://github.com/raghavan/PdfGptIndexer/tree/main)