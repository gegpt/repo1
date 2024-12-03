import streamlit as st
import os
import time
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# 确保文件夹存在，若不存在则创建
pdf_directory = 'pdfFiles'
vector_directory = 'vectorDB'
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(vector_directory, exist_ok=True)

# 设置 Hugging Face API Token (请确保已设置 Hugging Face 的 API 令牌)
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# 初始化 Streamlit 会话状态
if 'chat_template' not in st.session_state:
   st.session_state.chat_template = """I am a chatbot ， I only can find you answer in below content，
   if the answer is bad, please answer me try again.

   Context: {context}
   History: {history}

   User: {question}
   Chatbot:"""

if 'prompt_template' not in st.session_state:
    st.session_state.prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.chat_template,
    )

if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

# 使用 Hugging Face 的嵌入模型
if 'embeddings_store' not in st.session_state:
    st.session_state.embeddings_store = Chroma(
        persist_directory=vector_directory,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

# 使用 Hugging Face 的 GPT-2 模型
if 'language_model' not in st.session_state:
    model_id = "gpt2"  # 可以替换为您想使用的 Hugging Face 模型
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=300,  # 调整 max_length 来处理较长的输入
        max_new_tokens=100  # 设置生成的最大新 token 数量
    )
    st.session_state.language_model = HuggingFacePipeline(pipeline=hf_pipeline)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 主标题
st.title("Chatbot With PDFs")

# PDF 文件上传器
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# 显示聊天历史
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# 处理上传的 PDF 文件
if uploaded_file is not None:
    file_path = os.path.join(pdf_directory, uploaded_file.name)
    if not os.path.exists(file_path):
        with st.status("Saving file..."):
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            pdf_loader = PyPDFLoader(file_path)
            pdf_content = pdf_loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            document_sections = text_splitter.split_documents(pdf_content)
            st.session_state.embeddings_store = Chroma.from_documents(
                documents=document_sections,
                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
            st.session_state.embeddings_store.persist()

    # 启动 PDF 文档检索
    st.session_state.retriever = st.session_state.embeddings_store.as_retriever()

    # 启动问答链
    if 'qa_chain' not in st.session_state:
       st.session_state.qa_chain = RetrievalQA.from_chain_type(
           llm=st.session_state.language_model,
           chain_type='stuff',
           retriever=st.session_state.retriever,
           verbose=True,
           chain_type_kwargs={
               "verbose": True,
               "prompt": st.session_state.prompt_template,
               "memory": st.session_state.conversation_memory,
           }
       )

    # 处理用户输入并响应
    if user_input := st.chat_input("You:", key="user_input"):
       user_message = {"role": "user", "message": user_input}
       st.session_state.chat_history.append(user_message)
       with st.chat_message("user"):
           st.markdown(user_input)

       with st.chat_message("assistant"):
           with st.spinner("hold on. I am cooking..."):
               response = st.session_state.qa_chain.run(user_input)
           message_placeholder = st.empty()
           full_response = ""
           for chunk in response.split():
               full_response += chunk + " "
               time.sleep(0.05)
               # 添加光标来模拟打字效果
               message_placeholder.markdown(full_response + "▌")
           message_placeholder.markdown(full_response)

       chatbot_message = {"role": "assistant", "message": response}
       st.session_state.chat_history.append(chatbot_message)
else:
    st.write("請上傳一個 PDF 檔案來開始 ChatPDF.")
