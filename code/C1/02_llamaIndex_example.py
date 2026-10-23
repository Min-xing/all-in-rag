import os
# 如果 Hugging Face 下载速度慢，取消下面一行的注释，改用国内镜像
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 从 .env 文件中读取环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

# 方案一：使用 AIHubmix 的免费模型（当前已注释）
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",           # AIHubmix 平台上的模型名称
#     api_key=os.getenv("DEEPSEEK_API_KEY"), # 从环境变量读取密钥
#     api_base="https://aihubmix.com/v1",    # AIHubmix 的接口地址
#     is_chat_model=True                     # 使用对话接口
# )

# 方案二：使用 LlamaIndex 官方的 OpenAI 类（当前已注释）
# 注意：如果取消注释，需要先导入 from llama_index.llms.openai import OpenAI
# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 方案三：使用 OpenAI 兼容接口连接 DeepSeek（当前启用的方案）
Settings.llm = OpenAILike(
    model="deepseek-chat",                 # DeepSeek 对话模型名称
    temperature=0.7,                       # 随机性，越高回答越多样
    max_tokens=4096,                       # 单次生成的最大输出 token 数
    api_key=os.getenv("DEEPSEEK_API_KEY"), # 从环境变量读取 DeepSeek 密钥
    api_base="https://api.deepseek.com/v1",# DeepSeek 的 OpenAI 兼容接口地址
    is_chat_model=True,                    # 使用 chat 对话接口，而不是纯补全接口
    context_window=131072,                 # 模型上下文窗口大小（约 128K token）
)

# 设置嵌入模型：负责把文本转换成向量，供后续检索使用
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 读取指定 Markdown 文件，将其加载为文档对象
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 把文档切块、向量化，并建立向量索引
index = VectorStoreIndex.from_documents(docs)

# 基于索引创建查询引擎，内部包含“检索器”和“答案合成器”
query_engine = index.as_query_engine()

# 打印查询引擎当前使用的提示词模板，便于理解底层流程
print(query_engine.get_prompts())

# 发起提问：先从索引中检索相关内容，再交给大模型生成答案
print(query_engine.query("文中举了哪些例子?"))