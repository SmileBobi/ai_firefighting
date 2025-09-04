from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter #文本切分工具
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

import os
from dotenv import load_dotenv

from utils.load_documents import load_documents_from_folder

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")


# 1、初始化通义千问模型
llm = ChatTongyi(model_name="qwen-turbo",dashscope_api_key=api_key)

# 2、加载知识库：读取本地的pdf、word文档，作为知识数据源
documents = load_documents_from_folder("./docs")

# 3、将知识数据源进行切分：每300字符切一块，块之间重叠20个字符
# 这里是为了保证每块文本既不太大（方便处理），又有点上下文（避免断句）
#   1、FAQ : “问题+答案” 切分                     300   50
#   2、法律、合同、规章：按照“条款/小节”切分          500-800   100
#   3、技术文档、pdf：按“照段落、标题”切分           500   50
#   4、录音、聊天记录：按照“时间、对话轮次”切分
#   5、长文本：按照“固定的字数/tokens，重叠滑窗”切分
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# 4、创建嵌入模型，初始化向量工具，使用阿里的 DashScope API，把文本转成向量
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",  # DashScope专用嵌入模型
    dashscope_api_key=api_key  # 使用正确密钥
)

# 5、用 FAISS 创建一个向量库，把所有文本块存进去
# FAISS 负责做“相似度检索”，快速查找与用户提问最接近的内容块
db = FAISS.from_documents(texts, embeddings)

# 6、创建 RAG （检索增强生成）问答链
# RetrievalQA 就是LangChain 内置的 RAG 实现
# 先用 FAISS 检索最接近的文本块，再把文本块作为输入给 LLM 生成答案

qa = RetrievalQA.from_chain_type(
    llm = llm, # 大语言模型
    chain_type="stuff", # 简单的 RAG 模式，把检索到的文档直接拼接起来
    retriever=db.as_retriever() # 检索器，负责查找相似知识块
)

# 7、用户提问
query = "施工单位应当承担哪些消防施工的质量和安全责任？"

# 8、调用 QA 链：先检索，再生成答案
result = qa.invoke({"query":query})

# 9、打印答案
print(result["result"])