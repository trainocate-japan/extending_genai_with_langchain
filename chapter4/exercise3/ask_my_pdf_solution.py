import os
from dotenv import load_dotenv, find_dotenv
import tempfile
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.tracers.context import collect_runs
from langchain_pinecone import PineconeVectorStore
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
from langchain_classic.chains import create_history_aware_retriever
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from langsmith import Client

# API キーなどの設定
# python-dotenv を使用して、.env ファイルに記載された API キーを環境変数として設定する
load_dotenv(find_dotenv(), override=True)
os.environ.get('OPENAI_API_KEY')
os.environ.get('PINECONE_API_KEY')
os.environ.get('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Streamlit でのセッション情報の初期化
if 'costs' not in st.session_state:
    st.session_state.costs = []
if 'latest_run_id' not in st.session_state:
    st.session_state.latest_run_id = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Stremlit でのセッション ID の取得
ctx = get_script_run_ctx()

# CHECK
# Chat history の初期化
# ここでは Streamlit の機能を使って会話履歴を保持する StreamlitChatMessageHistory を使用します
chat_history = StreamlitChatMessageHistory(key="langchain_messages")

def init_page():
    st.set_page_config(
        page_title="Ask My PDF",
        page_icon="📖"
    )
    st.sidebar.title("Nav")

# PDF ファイルのアップロード
# st.file_uploader ウィジェットを使用してファイルをアップロードする
# load_document 関数と chunk_data 関数を呼び出して処理を実行する
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

# get_pdf_text から呼び出されて Document loader (PyPDFLoader) を実行
def load_document(file):
    name, extension = os.path.splitext(file.name)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    if extension == '.pdf':
        # TASK
        # 以下のパラメータで PyPDFLoader のインスタンスを作成し、ファイルをロードするコードを作成してください
        # file_path=tmp_file_path
        loader = PyPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        return data
    else:
        st.write('Document format is not supported!')
        return None

# get_pdf_text から呼び出されて Text splitter (RecursiveCharacterTextSplitter) を実行
def chunk_data(data, chunk_size=1024, chunk_overlap=256):
    # TASK
    # RecursiveCharacterTextSplitter のインスタンスを作成して、ファイルのデータをチャンクに分割するコードを作成してください
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Vector store の作成
# 具体的な処理は insert_or_fetch_embeddings 関数を呼び出して実行
def build_vector_store(chunks):
    index_name = "askadocument"
    vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)
    return vector_store

# build_vector_store から呼び出されて、インデックスの新規作成と Vector store の作成を行う
def insert_or_fetch_embeddings(index_name, chunks):
    pc = pinecone.Pinecone()
    # TASK
    # 以下のパラメータで Embedding model (OpenAIEmbeddings) のインスタンスを作成してください
    # model='text-embedding-3-small'
    # dimensions=1536
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
            # CHECK
            # Vector store (Pinecone) のインスタンスを作成
            vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    else:
        with st.spinner("Updating embeddings ... "):
            vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

# 既存のインデックスを削除
def delete_pinecone_index(index_name='all'):
    pc = pinecone.Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        for index in indexes:
            pc.delete_index(index)

# PDF ファイルをアップロードする画面を生成する関数
# 上記の各関数が呼び出されている
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
        # 既存のインデックスを削除するボタン
        # delete_pinecone_index 関数を呼び出して処理を実行
        st.markdown("## Delete Index")
        if st.button("Delete Index"):
            delete_pinecone_index()
            st.session_state.vector_store = None
            st.success("Vector store index deleted successfully!")

# ユーザーが画面上で選択したモデルのインスタンスを作成する
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-4o-mini", "GPT-4o"))
    if model == "GPT-4o-mini":
        st.session_state.model_name = "gpt-4o-mini"
    else:
        st.session_state.model_name = "gpt-4o"
    # TASK
    # 以下のパラメータで Chat model (ChatOpenAI) のインスタンスを作成してください
    # temperature=0
    # model=st.session_state.model_name
    model = ChatOpenAI(temperature=0, model=st.session_state.model_name)
    return model

# page_ask_my_pdf から呼び出され、RAG Chain を構成して実行する
def get_answer_with_history(model, vector_store, query, session_id='unused'):
    # TASK
    # 以下のパラメータで Retriever を作成してください
    # search_type='similarity'
    # search_kwargs={'k': 10}
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

    # TASK
    # 会話履歴を考慮してドキュメントの検索・収得を行う Retriever を作成してください
    # 組み込みの create_histroy_aware_retriever 関数を使用します
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    # Chat prompt の作成
    system_message = """以下の参考用のテキストの一部を参照して、質問に丁寧かつ親切に回答してください。もし参考用のテキストの中に回答に役立つ情報が含まれていなければ、分かりません、と答えてください。

    {context}"""
    human_message = "質問：{input}"

    # TASK
    # モデルに渡すプロンプトの Prompt template を作成してください
    # MessagePlaceholder を使用して chat_history が挿入されるようにしてください
    chat_prompt = ChatPromptTemplate.from_messages([

        (
            "system", system_message
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human", human_message
        )
    ])

    # Retriever による検索ドキュメントをコンテキストに追加
    add_context = RunnablePassthrough.assign(context=history_aware_retriever)

    # TASK
    # RAG の処理を実行する Chain を定義してください
    # add_context > chat_prompt > model > StrOutputParser の順に処理されるように定義します
    rag_chain = add_context | chat_prompt | model | StrOutputParser()

    runnable_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    with get_openai_callback() as cb:
        with collect_runs() as runs_cb:
            answer = runnable_with_history.invoke({'input': query}, config={"configurable": {"session_id": session_id}})
            run_id = runs_cb.traced_runs[0].id
            st.session_state.latest_run_id = run_id
    return answer, cb.total_cost

# チャット画面を生成する関数
def page_ask_my_pdf():
    st.title("📖 Ask My PDF(s)")

    """
    The messages are stored in Session State across re-runs automatically.
    You can view the contents of Session State in the expander below.
    """
    view_messages = st.expander("View the message contents in session state")

    # select_model 関数内で Chat model のインスタンスを作成して取得
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
                # 質問に対して回答を取得
                answer, cost = get_answer_with_history(model, vector_store, query, session_id=ctx.session_id)
            # 回答を表示
            st.chat_message("ai").write(answer)
            # コストを加算
            st.session_state.costs.append(cost)
            # フィードバック送信状態をリセット
            st.session_state.feedback_submitted = False
            # 最新の run_id を保持
            # st.session_state.latest_run_id = st.session_state.latest_run_id

        # フィードバックフォームの表示
        # streamlit_feedback でフォームを表示し、send_feedback 関数を呼び出してフィードバックを送信
        if st.session_state.get("latest_run_id") and not st.session_state.feedback_submitted:
            run_id = st.session_state.latest_run_id
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=send_feedback,
                key=f"fb_k_{run_id}",
                args=[run_id]
            )

    # Draw the messages at the end, so newly generated ones show up immediately
    with view_messages:
        """
        Message History initialized with:
        chat_history = StreamlitChatMessageHistory(key="langchain_messages")
        
        Contents of `st.session_state.langchain_messages`:
        """
        view_messages.json(st.session_state.langchain_messages)

# LangSmith にフィードバックを送信する関数
def send_feedback(user_feedback, run_id):
    scores = {"👍": 1, "👎": 0}
    score_key = user_feedback["score"]
    score = scores[score_key]
    comment = user_feedback.get("text")

    # LangSmith API でフィードバックを送信
    client = Client()
    client.create_feedback(
        run_id=run_id,
        key="thumbs",
        score=score,
        comment=comment,
    )

    # フィードバック送信が完了したことを記録
    st.session_state.feedback_submitted = True
    st.success("Thank you for your feedback!")

# main 関数
# まず最初に実行されてアプリケーションの画面全体の骨格を構成する
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
