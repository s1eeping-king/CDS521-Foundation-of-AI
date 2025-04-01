from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from werkzeug.utils import secure_filename
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import json
import numpy as np
import time
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import dashscope
from http import HTTPStatus

# 加载环境变量
load_dotenv()

# 添加文件记录相关的常量
PROCESSED_FILES_RECORD = 'processed_files.json'

class AliyunEmbeddings(Embeddings):
    def __init__(self, api_key: str, max_retries: int = 3, batch_size: int = 5):
        self.api_key = api_key
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_text_length = 8192  # 将限制调小，给API留出安全余量
        dashscope.api_key = api_key

    def _sanitize_text(self, text: str) -> str:
        """清理文本中的无效Unicode字符，特别是代理对字符"""
        # 替换或删除无效的代理对字符
        cleaned_text = ""
        for char in text:
            # 检查是否是代理对字符（surrogate pair characters）
            if 0xD800 <= ord(char) <= 0xDFFF:
                # 替换为Unicode替换字符或空字符串
                cleaned_text += "�"  # 使用Unicode替换字符
            else:
                cleaned_text += char
        return cleaned_text
        
    def _truncate_text(self, text: str) -> str:
        """截断过长的文本并清理无效Unicode字符"""
        # 首先清理文本
        text = self._sanitize_text(text)
        
        # 计算文本字节长度（因为API可能是按字节计算）
        try:
            text_bytes_length = len(text.encode('utf-8'))
            text_chars_length = len(text)
            
            if text_bytes_length > self.max_text_length or text_chars_length > self.max_text_length:
                print(f"警告：文本长度 {text_chars_length} 字符/{text_bytes_length} 字节超过限制 {self.max_text_length}，将被截断")
                # 确保截断后的文本不超过最大长度
                truncated = text
                while len(truncated.encode('utf-8')) > self.max_text_length:
                    truncated = truncated[:int(len(truncated) * 0.9)]  # 逐步截断到合适大小
                return truncated
            return text
        except UnicodeEncodeError as e:
            print(f"编码错误：{str(e)}，尝试更严格的字符清理")
            # 使用更激进的方法：只保留ASCII字符
            text = ''.join(char for char in text if ord(char) < 128)
            return text[:self.max_text_length]

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                # 截断过长的文本
                truncated_texts = [self._truncate_text(text) for text in texts]
                
                # 检查每个文本是否超过长度限制
                for i, text in enumerate(truncated_texts):
                    bytes_len = len(text.encode('utf-8'))
                    if bytes_len > self.max_text_length:
                        print(f"警告：第{i+1}个文本截断后仍然超出限制（{bytes_len} > {self.max_text_length}）")
                
                # 将文本列表转换为dashscope需要的格式
                input_data = [{'text': text} for text in truncated_texts]
                
                # 调用dashscope API
                resp = dashscope.MultiModalEmbedding.call(
                    model="multimodal-embedding-v1",
                    input=input_data
                )
                
                if resp.status_code == HTTPStatus.OK:
                    # 从响应中提取embedding
                    embeddings = [item['embedding'] for item in resp.output['embeddings']]
                    return embeddings
                else:
                    raise Exception(f"API调用失败: {resp.code} - {resp.message}")
                    
            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retrying

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs in batches."""
        embeddings = []
        # 使用tqdm显示总体进度
        with tqdm(total=len(texts), desc="生成文本向量", unit="chunk") as pbar:
            # 批量处理文本
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._embed_with_retry(batch)
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed_with_retry([text])[0]

app = Flask(__name__)

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-key",
    http_client=None  # Disable default http client to avoid proxies error
)

# Initialize embeddings with Aliyun API
embeddings = AliyunEmbeddings(
    api_key="sk-547da51ddc224af08338396623eb0348",
    batch_size=1  # 减小批处理大小，避免总大小超限
)

# Configure upload folder
UPLOAD_FOLDER = 'knowledge_base'
ALLOWED_EXTENSIONS = {
    'txt', 'md', 'json', 'pdf', 'docx', 'doc', 
    'xlsx', 'xls', 'pptx', 'ppt'
}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize vector store
vector_store = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_loader(file_path):
    """根据文件类型返回适当的文档加载器"""
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return PyPDFLoader(file_path)
    elif file_extension in ['docx', 'doc']:
        return Docx2txtLoader(file_path)
    elif file_extension in ['xlsx', 'xls']:
        return UnstructuredExcelLoader(file_path)
    elif file_extension in ['pptx', 'ppt']:
        return UnstructuredPowerPointLoader(file_path)
    else:
        # 对于文本文件，尝试不同的编码
        try:
            return TextLoader(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            return TextLoader(file_path, encoding='gbk')

def initialize_vector_store():
    global vector_store
    print("正在初始化向量存储...")
    if os.path.exists('vector_store'):
        print("找到现有的向量存储，正在加载...")
        vector_store = FAISS.load_local('vector_store', embeddings, allow_dangerous_deserialization=True)
        print("向量存储加载成功")
    else:
        print("未找到现有的向量存储")
        vector_store = None

def load_processed_files():
    """加载已处理文件的记录"""
    if os.path.exists(PROCESSED_FILES_RECORD):
        try:
            with open(PROCESSED_FILES_RECORD, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载处理记录时出错: {str(e)}")
            return {}
    return {}

def save_processed_files(record):
    """保存已处理文件的记录"""
    try:
        with open(PROCESSED_FILES_RECORD, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存处理记录时出错: {str(e)}")

def get_file_hash(file_path):
    """获取文件的哈希值，用于检测文件是否被修改"""
    try:
        stat = os.stat(file_path)
        return f"{stat.st_size}_{stat.st_mtime}"
    except Exception as e:
        print(f"获取文件哈希值时出错: {str(e)}")
        return None

def update_vector_store():
    global vector_store
    documents = []
    print("开始更新向量存储...")
    
    # 加载已处理文件的记录
    processed_files = load_processed_files()
    
    # 获取当前目录中的所有文件
    current_files = {}
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_hash = get_file_hash(file_path)
            if file_hash:
                current_files[filename] = file_hash
    
    # 检查是否有新文件或修改过的文件
    new_or_modified_files = []
    for filename, file_hash in current_files.items():
        if filename not in processed_files or processed_files[filename] != file_hash:
            new_or_modified_files.append(filename)
    
    if not new_or_modified_files:
        print("没有发现新文件或修改过的文件，向量存储是最新的")
        return
    
    print(f"发现 {len(new_or_modified_files)} 个新文件或修改过的文件:")
    for filename in new_or_modified_files:
        print(f"- {filename}")
    
    # 处理新文件或修改过的文件
    for filename in new_or_modified_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        size = os.path.getsize(file_path)
        print(f"\n正在处理文件: {filename} ({size / (1024*1024):.2f} MB)")
        
        try:
            loader = get_file_loader(file_path)
            print(f"使用加载器: {loader.__class__.__name__}")
            
            # 使用更小的分块大小，确保不超过API限制
            chunk_size = 2500  # 设置一个更安全的分块大小
            chunk_overlap = 100
            
            # 根据文件类型显示不同的加载进度
            if filename.lower().endswith('.pdf'):
                print("正在加载PDF文件...")
                docs = loader.load()
                print(f"PDF加载完成，包含 {len(docs)} 页")
            else:
                with tqdm(total=size, unit='B', unit_scale=True, desc=f"加载文件 {filename}") as pbar:
                    docs = loader.load()
                    pbar.update(size)
            
            print(f"文件 {filename} 加载成功，包含 {len(docs)} 个文档")
            
            # 对所有文档进行分块处理
            with tqdm(total=len(docs), desc=f"分块处理 {filename}") as pbar:
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                docs = text_splitter.split_documents(docs)
                pbar.update(len(docs))
            print(f"文件被分割成 {len(docs)} 个文本块")
            
            documents.extend(docs)
            
            # 更新处理记录
            processed_files[filename] = current_files[filename]
            
        except Exception as e:
            print(f"加载文件 {filename} 时出错: {str(e)}")
            continue
    
    if documents:
        print(f"\n总共加载了 {len(documents)} 个文档")
        try:
            if vector_store is None:
                print("创建新的向量存储...")
                vector_store = FAISS.from_documents(documents, embeddings)
            else:
                print("将新文档添加到现有向量存储...")
                vector_store.add_documents(documents)
            
            print("正在保存向量存储...")
            vector_store.save_local('vector_store')
            save_processed_files(processed_files)
            print("向量存储已更新并保存")
        except Exception as e:
            print(f"创建向量存储时出错: {str(e)}")
            vector_store = None
    else:
        print("没有找到任何文档")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # 保存文件
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # 更新向量存储
            update_vector_store()
            
            return jsonify({'message': '文件上传成功'})
        except Exception as e:
            return jsonify({'error': f'文件处理失败: {str(e)}'}), 500
    else:
        return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    conversation_id = data.get('conversation_id')
    history = data.get('history', [])
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Get relevant context from knowledge base
        context = ""
        print(f"向量存储状态: {'已初始化' if vector_store else '未初始化'}")
        if vector_store:
            print("正在搜索相关文档...")
            docs = vector_store.similarity_search(question, k=6)
            
            # 收集文档内容和来源
            doc_contents = []
            doc_sources = []
            for doc in docs:
                # 从文档的metadata中获取文件名
                source = doc.metadata.get('source', '未知来源')
                # 只保留文件名，去掉路径
                source = os.path.basename(source)
                doc_contents.append(doc.page_content)
                doc_sources.append(source)
            
            context = "\n".join(doc_contents)
            print(f"找到 {len(docs)} 个相关文档片段")
            print("相关文档内容:")
            for i, (content, source) in enumerate(zip(doc_contents, doc_sources)):
                print(f"文档 {i+1} ({source}): {content[:1000]}...")
        else:
            print("警告: 向量存储未初始化，无法检索相关文档")
        
        # Add context to the system message
        system_message = {
            "role": "system",
            "content": f"""
            You are a professional family doctor AI consultant and need to follow the following rules:

            Knowledge source: Only use knowledge base knowledge ```{context}``` and do not allow external knowledge to be called.
            Role setting: As an experienced general practitioner, specializing in chronic disease management, symptom screening, and health guidance.
            Response requirements:
            Provide explanations and recommendations based on authoritative literature (such as the "Chinese Guidelines for the Prevention and Treatment of Hypertension").
            Label the names of the literature used.
            
            Knowledge sources used:
            {', '.join(doc_sources) if 'doc_sources' in locals() else 'No specific sources available'}
            """
        }
        
        # Add system message to the beginning of the conversation
        messages = [system_message] + history
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": request.host_url,
                "X-Title": "OpenRouter Chat App",
            },
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages,
            temperature=0.7,
        )
        
        answer = completion.choices[0].message.content
        
        return jsonify({
            "answer": answer,
            "conversation_id": conversation_id,
            "sources": doc_sources if 'doc_sources' in locals() else []
        })
    
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/files')
def list_files():
    try:
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"知识库目录: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"目录内容: {os.listdir(UPLOAD_FOLDER)}")
    
    # 初始化向量存储
    initialize_vector_store()
    
    # 更新向量存储（只处理新文件或修改过的文件）
    update_vector_store()
    
    app.run(debug=True)