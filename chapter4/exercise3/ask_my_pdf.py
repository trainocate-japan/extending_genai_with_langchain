import os
from dotenv import load_dotenv, find_dotenv
import tempfile
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.vectorstores.pinecone import Pinecone
import pinecone
from pinecone import ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

# API キーなどを設定
# python-dotenv を使用して、.env ファイルに記載された API キーを環境変数として設定する
load_dotenv(find_dotenv(), override=True)
os.environ.get('OPENAI_API_KEY')
os.environ.get('PINECONE_API_KEY')
os.environ.get('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Stremlit でのセッション ID を取得する
ctx = get_script_run_ctx()

# Chat history を初期化する
chat_history = StreamlitChatMessageHistory(key="langchain_messages")

def init_page():
    st.set_page_config(
        page_title="Ask My PDF",
        page_icon="📖"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []

# ユーザーが画面上で選択したモデルのインスタンスを作成する
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-4o-mini", "GPT-4o"))
    if model == "GPT-4o-mini":
        st.session_state.model_name = "gpt-4o-mini"
    else:
        st.session_state.model_name = "gpt-4o"

    model = ChatOpenAI(temperature=0, model=st.session_state.model_name)
    return model

def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',
        type='pdf',
        accept_multiple_files=True
    )
    if uploaded_file:
        if isinstance(uploaded_file, list):
            data_list = []
            for file in uploaded_file:
                data = load_document(file)
                if data:
                    data_list.extend(data)
            if data_list:
                chunks = chunk_data(data_list)
                return chunks
        else:
            data = load_document(uploaded_file)
            if data:
                chunks = chunk_data(data)
                return chunks
    return None

# get_pdf_text から呼び出されて Document loader (PyPDFLoader) を実行する
def load_document(file):
    name, extension = os.path.splitext(file.name)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    if extension == '.pdf':
        loader = PyPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        return data
    else:
        st.write('Document format is not supported!')
        return None

# get_pdf_text から呼び出されて Text splitter (RecursiveCharacterTextSplitter) を実行
def chunk_data(data, chunk_size=256, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def build_vector_store(chunks):
    index_name = "askadocument"
    vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)
    return vector_store

# build_vector_store から呼び出されて、インデックスの新規作成と Vector store の作成を行う
def insert_or_fetch_embeddings(index_name, chunks):
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Create or update the index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        with st.spinner("Creating index and embeddings ... "):
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    else:
        with st.spinner("Updating embeddings ... "):
            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

# insert_or_fetch_embeddings から呼び出されて既存のインデックスを削除する
def delete_pinecone_index(index_name='all'):
    pc = pinecone.Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        for index in indexes:
            pc.delete_index(index)

# PDF ファイルをアップロードする画面
def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        st.markdown("## Upload PDF files")
        chunks = get_pdf_text()
        if chunks:
            with st.spinner("Loading PDF ..."):
                vector_store = build_vector_store(chunks)
                st.session_state.vector_store = vector_store
        st.markdown("## Delete Index")
        if st.button("Delete Index"):
            delete_pinecone_index()
            st.session_state.vector_store = None
            st.success("Vector store index deleted successfully!")

# page_ask_my_pdf から呼び出され、RAG Chain を構成して実行する
def get_answer_with_history(model, vector_store, query, session_id='unused'):
    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    # Chat prompt
    system_message = """以下の参考用のテキストの一部を参照して、質問に回答してください。もし参考用のテキストの中に回答に役立つ情報が含まれていなければ、分からない、と答えてください。

    {context}"""
    human_message = "質問：{input}"

    chat_prompt = ChatPromptTemplate.from_messages([

        (
            "system", system_message
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human", human_message
        )
    ])

    # Create a runnable that passes inputs added with context
    add_context = RunnablePassthrough.assign(context=history_aware_retriever)

    # Create a Chain
    rag_chain = add_context | chat_prompt | model | StrOutputParser()

    runnable_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    with get_openai_callback() as cb:
        answer = runnable_with_history.invoke({'input': query}, config={"configurable": {"session_id": session_id}})
    return answer, cb.total_cost

# チャット画面
def page_ask_my_pdf():
    st.title("📖 Ask My PDF(s)")

    """
A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation.
The messages are stored in Session State across re-runs automatically. You can view the contents of Session State
in the expander below. 
"""
    view_messages = st.expander("View the message contents in session state")

    model = select_model()

    if "vector_store" in st.session_state:
        vector_store = st.session_state.vector_store

    else:
        vector_store = None
    
    if vector_store:
        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)
        if query := st.chat_input():
            st.chat_message("human").write(query)
            with st.spinner("ChatGPT is typing ..."):
                answer, cost = get_answer_with_history(model, vector_store, query, session_id=ctx.session_id)
            st.chat_message("ai").write(answer)
            st.session_state.costs.append(cost)
    # Draw the messages at the end, so newly generated ones show up immediately
    with view_messages:
        """
        Message History initialized with:
        ```python
        chat_history = StreamlitChatMessageHistory(key="langchain_messages")
        ```

        Contents of `st.session_state.langchain_messages`:
        """
        view_messages.json(st.session_state.langchain_messages)

def main():
    init_page()

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()