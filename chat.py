import os
from openai import OpenAI
from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from create_kb import *

# 定义常量
DB_PATH = "VectorStore"  # 向量数据库存储路径
TMP_NAME = "tmp_abcd"    # 临时知识库名称

# 配置DashScope嵌入模型
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

# 若使用本地嵌入模型，请取消以下注释：
# from langchain_community.embeddings import ModelScopeEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding
# embeddings = ModelScopeEmbeddings(model_id="iic/nlp_gte_sentence-embedding_chinese-large")
# EMBED_MODEL = LangchainEmbedding(embeddings)

# 参考文档 https://www.modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large

# 设置全局嵌入模型
Settings.embed_model = EMBED_MODEL

def get_model_response(multi_modal_input, history, model, temperature, max_tokens, history_round, db_name, similarity_threshold, chunk_cnt):
    """
    获取模型响应的主函数
    Args:
        multi_modal_input: 多模态输入（包含文本和文件）
        history: 对话历史
        model: 使用的模型名称
        temperature: 温度参数
        max_tokens: 最大生成token数
        history_round: 历史对话轮数
        db_name: 知识库名称
        similarity_threshold: 相似度阈值
        chunk_cnt: 检索的文本块数量
    """
    # 获取用户输入的文本
    prompt = history[-1][0]
    tmp_files = multi_modal_input['files']
    
    # 处理临时知识库
    if os.path.exists(os.path.join("File",TMP_NAME)):
        db_name = TMP_NAME
    else:
        if tmp_files:
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME
            
    print(f"prompt:{prompt},tmp_files:{tmp_files},db_name:{db_name}")
    
    try:
        # 初始化重排序器
        dashscope_rerank = DashScopeRerank(top_n=chunk_cnt,return_documents=True)
        
        # 加载向量索引
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(DB_PATH,db_name)
        )
        index = load_index_from_storage(storage_context)
        print("index获取完成")
        
        # 配置检索器
        retriever_engine = index.as_retriever(
            similarity_top_k=20,
        )
        
        # 检索相关文本块
        retrieve_chunk = retriever_engine.retrieve(prompt)
        print(f"原始chunk为：{retrieve_chunk}")
        
        # 尝试对检索结果进行重排序
        try:
            results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=prompt)
            print(f"rerank成功，重排后的chunk为：{results}")
        except:
            results = retrieve_chunk[:chunk_cnt]
            print(f"rerank失败，chunk为：{results}")
            
        # 构建上下文信息
        chunk_text = ""
        chunk_show = ""
        for i in range(len(results)):
            if results[i].score >= similarity_threshold:
                chunk_text = chunk_text + f"## {i+1}:\n {results[i].text}\n"
                chunk_show = chunk_show + f"## {i+1}:\n {results[i].text}\nscore: {round(results[i].score,2)}\n"
        print(f"已获取chunk：{chunk_text}")
        
        # 构建提示模板
        prompt_template = f"请参考以下内容：{chunk_text}，以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"
    except Exception as e:
        print(f"异常信息：{e}")
        prompt_template = prompt
        chunk_show = ""
        
    # 准备对话历史
    history[-1][-1] = ""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )                
    
    # 构建消息列表
    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}
    messages = []
    history_round = min(len(history),history_round)
    for i in range(history_round):
        messages.append({'role': 'user', 'content': history[-history_round+i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round+i][1]})
    messages.append({'role': 'user', 'content': prompt_template})
    messages = [system_message] + messages
    
    # 调用模型生成回答
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
        )
        
    # 流式返回响应
    assistant_response = ""
    for chunk in completion:
        assistant_response += chunk.choices[0].delta.content
        history[-1][-1] = assistant_response
        yield history,chunk_show