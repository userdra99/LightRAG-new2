# LightRAG-vLLM-Streamlit

A multi-service application that combines LightRAG for knowledge graph-based retrieval-augmented generation with vLLM for high-performance inference and Streamlit for a user-friendly web interface.

## 🚀 Features

- **Knowledge Graph RAG**: Uses LightRAG to create and query knowledge graphs from documents
- **High-Performance Inference**: Leverages vLLM for fast LLM and embedding model serving
- **Web Interface**: Streamlit-based UI for document upload and querying
- **Multi-Format Support**: Processes PDF, TXT, DOCX, XLSX, and CSV files
- **Real-time Processing**: Automatic document processing and indexing
- **Service Monitoring**: Built-in health checks and status monitoring

## 🏗️ Architecture

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │ Streamlit UI │ │ LightRAG │ │ vLLM │ │ (Port 8501) │◄──►│ Processor │◄──►│ LLM (8000) │ │ │ │ │ │ Embed (8001) │ └─────────────────┘ └─────────────────┘ └─────────────────┘

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/LightRAG-vLLM-Streamlit.git
   cd LightRAG-vLLM-Streamlit

Create virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Create required directories:
mkdir -p DOCSSOURCE logs data

🚦 Usage

Quick Start
Start all services:
./start.sh

Access the web interface: Open your browser and go to
http://localhost:8501

Upload documents:
Use the "Documents" tab to upload PDF, TXT, DOCX, XLSX, or CSV files
Documents are automatically processed and indexed

Query your documents:
Go to the "Query" tab
Enter your question and select a query mode
Get AI-powered answers based on your documents

Manual Service Management
Start individual services:
# Start vLLM LLM service
venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16 \
    --port 8000 --host 0.0.0.0

# Start vLLM Embedding service
venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model jinaai/jina-embeddings-v4-vllm-retrieval \
    --port 8001 --host 0.0.0.0 --task embedding

# Start LightRAG processor
venv/bin/python lightrag_processor.py

# Start Streamlit UI
venv/bin/streamlit run lightrag_app.py
Stop all services:
./stop.sh

⚙️ Configuration
Models

Edit
config/vllm_config.py
to change models:
class VLLMConfig:
    LLM_MODEL = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"
    EMBED_MODEL = "jinaai/jina-embeddings-v4-vllm-retrieval"
    LLM_PORT = 8000
    EMBED_PORT = 8001

LightRAG Settings
Edit
config/lightrag_config.py
for document processing:
class LightRAGConfig:
    WORKING_DIR = "./data"
    DOCSSOURCE_PATH = "./DOCSSOURCE"
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 100
    SUPPORTED_FORMATS = [".pdf", ".txt", ".docx", ".xlsx", ".csv"]

GPU Memory
Adjust GPU memory utilization in
start.sh
:
--gpu-memory-utilization 0.6  # For LLM service
--gpu-memory-utilization 0.3  # For embedding service

📊 Query Modes
Hybrid: Combines local and global knowledge retrieval
Local: Focuses on specific document chunks
Global: Uses high-level knowledge graph structure
Naive: Simple text-based retrieval

🔧 Troubleshooting

Common Issues
GPU Memory Error:
Reduce
--gpu-memory-utilization
values in
start.sh

Use smaller models in
vllm_config.py

Service Connection Failed:
Check if ports 8000, 8001, 8501 are available
Verify vLLM services are running:
curl http://localhost:8000/v1/models

LightRAG Not Starting:
Check logs:
tail -f logs/lightrag.log

Ensure vLLM services are running before starting LightRAG

Logs
Monitor service logs:
# View all logs
tail -f logs/*.log

# Specific service logs
tail -f logs/vllm_llm.log
tail -f logs/vllm_embedding.log
tail -f logs/lightrag.log
tail -f logs/streamlit.log

📁 Project Structure
├── config/
│   ├── lightrag_config.py      # LightRAG configuration
│   └── vllm_config.py          # vLLM model configuration
├── data/                       # Processed data storage
├── DOCSSOURCE/                 # Source documents
├── logs/                       # Service logs
├── services/                   # Service modules
├── utils/                      # Utility functions
├── lightrag_app.py            # Streamlit web interface
├── lightrag_processor.py      # Document processing service
├── main.py                    # Main application entry
├── start.sh                   # Service startup script
├── stop.sh                    # Service shutdown script
└── requirements.txt           # Python dependencies

🤝 Contributing
Fork the repository
Create a feature branch (
git checkout -b feature/amazing-feature
)
Commit your changes (
git commit -m 'Add amazing feature'
)
Push to the branch (
git push origin feature/amazing-feature
)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
LightRAG - Knowledge graph-based RAG framework
vLLM - High-throughput LLM inference
Streamlit - Web app framework

📧 Support
For questions and support, please open an issue in the GitHub repository.
Note: This application requires a CUDA-compatible GPU for optimal performance. CPU-only inference is possible but significantly slower.
