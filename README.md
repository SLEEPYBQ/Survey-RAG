# Survey-RAG

Survey-RAG is a tool for processing academic survey PDF documents and extracting information using large language models. This tool utilizes vector databases and Retrieval-Augmented Generation (RAG) to efficiently extract structured information from multiple PDF files. It employs a three-stage process:

1. **Embedding Generation**: Extract text from PDFs and generate embeddings
2. **Retrieval**: Query the embeddings with specific questions
3. **Analysis**: Process and consolidate the results

## Features

- PDF text extraction and chunking for better context preservation
- Vector embedding generation using OpenAI or Azure OpenAI APIs
- Improved retrieval strategy using Maximum Marginal Relevance (MMR)
- Parallel processing for efficient handling of multiple documents
- Structured output in CSV format (individual and consolidated results)
- Predefined question set focused on extracting specific information from research papers

## Project Structure

- **main.py**: Entry point of the application with command-line interface
- **embedding.py**: Functions for PDF text extraction and embedding generation
- **retrieval.py**: Functions for querying document content using LLMs
- **utils.py**: Helper functions, predefined questions, and output formatting

## Prerequisites

- Python 3.8+
- OpenAI API key or Azure OpenAI API key
- Required Python packages (install via `pip install -r requirements.txt`):
  - langchain
  - langchain_openai
  - langchain_community
  - textract
  - tqdm
  - faiss-cpu
  - openai

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your API keys for OpenAI or Azure OpenAI

## Usage

### Basic Usage

```bash
python main.py --input_dir ./pdfs --output_dir ./results --api_type openai --api_key YOUR_API_KEY
```

### Command-line Arguments

```
Arguments:
  --input_dir INPUT_DIR     PDF files input directory
  --output_dir OUTPUT_DIR   Embedding and results output directory
  
  --api_type {openai,azure} API type: openai or azure (default: openai)
  
  # OpenAI options
  --api_base API_BASE       OpenAI API base URL (default: https://api.openai-proxy.org/v1)
  --api_key API_KEY         OpenAI API key
  
  # Azure OpenAI options
  --api_version API_VERSION Azure OpenAI API version (default: 2023-05-15)
  --api_endpoint API_ENDPOINT Azure OpenAI endpoint
  --api_key_azure API_KEY_AZURE Azure OpenAI API key
  
  --max_workers MAX_WORKERS Max number of parallel workers (default: CPU count)
  
  # Mode options
  --mode {process,query,both} Run mode: process=PDF processing only, query=query only, both=process and query (default: both)
  --question_id QUESTION_ID Question ID to query (use to query a single question)
  --all_questions          Query all questions sequentially
  --consolidated_csv       Generate a consolidated CSV with all results (default: True)
```

### Example Workflows

#### Process PDFs and Generate Embeddings

```bash
python main.py --input_dir ./pdfs --output_dir ./results --mode process
```

#### Query Embeddings for a Specific Question

```bash
python main.py --output_dir ./results --mode query --question_id robot_name
```

#### Query All Questions and Generate Consolidated Results

```bash
python main.py --output_dir ./results --mode query --all_questions
```

#### Process PDFs and Query All Questions

```bash
python main.py --input_dir ./pdfs --output_dir ./results --mode both --all_questions
```

#### Distinguishing between OpenAI and HKUST ITSC Azure
You can specify the API type in the command line. For example:
But you have to define the API key in the `main.py` file.
1. **OpenAI**:
   ```bash
   python main.py --input_dir ./pdfs --output_dir ./results --mode both --all_questions --api_type openai
   ```

2. **HKUST ITSC Azure**:
   ```bash
   python main.py --input_dir ./pdfs --output_dir ./results --mode both --all_questions --api_type azure
   ```


### Predefined Questions

The system comes with predefined questions targeting specific aspects of research papers, including:

- Stakeholder information
- Sample size and demographic details
- Robot types and functionalities
- User engagement, acceptance, and trust
- Study methodology and testing context
- Key findings and additional information

## How It Works

1. **PDF Processing**:
   - PDFs are processed in parallel for efficiency
   - Text is extracted and split into chunks with overlap for context preservation
   - Embeddings are generated for each chunk and stored using FAISS

2. **Retrieval**:
   - Questions are processed against the embeddings
   - The system uses Maximum Marginal Relevance (MMR) to balance relevancy and diversity
   - An LLM (default: gpt-4o-mini) generates answers based on retrieved context

3. **Results Consolidation**:
   - Individual CSV files are created for each question
   - Optional consolidated CSV combines all results for easier analysis

![Untitled-2023-06-16-1537](https://github.com/raghavan/PdfGptIndexer/assets/131585/2e71dd82-bf4f-44db-b1ae-908cbb465deb)


## Customization

### Adding Custom Questions

You can modify the `get_questions()` function in `utils.py` to add, remove, or modify questions.

### Adjusting Retrieval Parameters

You can adjust the retrieval parameters in `retrieval.py`:
- `search_type`: Type of search algorithm (default: "mmr")
- `k`: Number of documents to retrieve (default: 5)
- `fetch_k`: Number of initial candidates to fetch (default: 10)
- `lambda_mult`: Diversity parameter for MMR (default: 0.7)

## License

[Specify your license here]

## Acknowledgements
This project was inspired by and reuses code from the following projects:
1. [PdfGptIndexer](https://github.com/raghavan/PdfGptIndexer/tree/main)
