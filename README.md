# CDS521-Foundation-of-AI

A Flask-based intelligent medical Q&A system integrated with Aliyun embedding models and OpenRouter LLM APIs, supporting multi-format document management for healthcare professionals.

## Key Features
- **Multi-format Support**: Handles 9 file formats (PDF/DOCX/XLSX/PPT etc.) using document loaders like `PyPDFLoader`
- **Smart Semantic Search**: FAISS vector database with dynamic update capability
- **Clinical Decision Support**: Provides evidence-based answers referencing guidelines like *Chinese Guidelines for Hypertension Prevention*
- **Conversation Management**: Maintains dialogue context with conversation IDs
- **Version Control**: Detects file modifications through SHA256 hashing

## Environment
```bash
Python 3.8+ | Flask | OpenAI | LangChain | DashScope | python-dotenv
```

## Quick Start
1. **Install dependencies**
```bash
pip install flask openai langchain dashscope python-dotenv tqdm
```

2. **Configure environment**
```env
# .env
ALIYUN_API_KEY=your_aliyun_key
OPENROUTER_API_KEY=your_openrouter_key
```

3. **Launch service**
```bash
python main.py
```

## API Endpoints
| Endpoint | Method | Functionality | Example Request |
|----------|--------|----------------|-----------------|
| `/upload` | POST | File upload | `curl -F "file=@medical.pdf" http://localhost:5000/upload` |
| `/ask` | POST | Q&A interaction | `{"question": "Hypertension medication", "history": []}` |
| `/files` | GET | List knowledge base | Returns JSON file list |

## Document Processing
1. Upload files to `knowledge_base` directory
2. Automatic detection of new/modified files
3. Text chunking (2500 chars/chunk, 100 overlap)
4. Generate embeddings via Aliyun Multimodal API
5. Build FAISS vector index

## Architecture Highlights
1. **Multi-layer Security**  
   - Input sanitization for special characters
   - API request retry mechanism (max 3 attempts)
2. **Performance Optimization**  
   - Batch processing (default batch_size=5)
   - Progress tracking with tqdm
3. **Medical Specificity**  
   - Specialized embeddings for clinical terminology
   - Context-aware response generation

## Notes
1. First launch creates `vector_store` directory automatically
2. Supported formats: TXT/PDF/DOCX/XLSX/PPT/MD/JSON
3. Response time averages 2-3s per query (tested on 4GB RAM)
4. Debug mode recommended with `batch_size=1`

