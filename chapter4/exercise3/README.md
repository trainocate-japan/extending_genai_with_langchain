# 演習の準備
1. Chrome ブラウザで Google アカウントにログインしていることを確認した上で、以下のリンクを右クリックししてシークレットウィンドウで開きます。  
   https://shell.cloud.google.com
   
3. 開いた Cloud Shell に以下のコマンドをコピー＆ペーストで入力して実行します。
   ```
   git clone https://github.com/trainocate-japan/extending_genai_with_langchain.git
   cd ./extending_genai_with_langchain/chapter4/exercise3
   ```
   
4. 以下のコマンドを実行して Python の仮想環境を作成します。
   ```
   python -m venv venv
   source venv/bin/activate
   ```
   
5. 以下のコマンドを実行して必要な Python パッケージをインストールします。
   ```
   pip install -q -r requirements.txt
   ```

6. 画面上部の [**エディタを開く**] アイコンをクリックします。
7. 画面左上のメニューアイコンをクリックし、[**File**] > [**Open Folder**] をクリックします。
8. `/homne/YOUR_NAME/extending_genai_with_langchain/chapter4/exercise3` と入力し [**OK**] をクリックします。(YOUR_NAME の部分はご自身のアカウント名になります)

9. `exercise3` ディレクトリに `.env` という名前のファイルを作成して、以下のように API キーを保存します。(＝の右側には実際のキーの値を入力してください)
   ```
   OPENAI_API_KEY=
   LANGCHAIN_API_KEY=
   PINECONE_API_KEY=
   ```
