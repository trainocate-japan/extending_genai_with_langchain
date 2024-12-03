import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

# API キーなどの設定
# python-dotenv を使用して、.env ファイルに記載された API キーを環境変数として設定する
load_dotenv(find_dotenv(), override=True)
os.environ.get('OPENAI_API_KEY')
os.environ.get('PINECONE_API_KEY')
os.environ.get('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# FastAPI のインスタンスを作成
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# 1 つめのエンドポイント
# ChatOpenAI のインスタンスを直接実行
add_routes(
    app,
    ChatOpenAI(model="gpt-4o-mini"),
    path="/openai",
)

# 2 つめのエンドポイント
# 入力されたワードについてのジョークを作成してくれる簡単な Chain を実行
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)