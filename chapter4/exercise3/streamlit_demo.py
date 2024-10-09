import streamlit as st

# Streamlit ではコードが処理される上から順に Web 画面上でウィジェット(部品)が表示される。
st.title('Streamlit Demo')
# st.caption('これは Streamlit のデモアプリケーションです')
# st.subheader('アプリの概要')
# st.text('これは Streamlit の概要を簡単に紹介するためのアプリケーションです。\n'
#         'コードが処理される上から順に、このようにウィジェットが表示されていきます。')

# # サイドナビゲーション
# st.sidebar.title('Navigation')

# # コードスニペット
# code = '''
# import streamlit as st
# st.title('Streamlit Demo')
# '''
# st.text('コードも表示できます。')
# st.code(code, language='python')

# # テキストボックス
# name = st.text_input('名前')
# # ボタン
# submit_btn = st.button('Submit')
# cancel_btn = st.button('Cancel')

# if submit_btn:
#     st.text(f'ようこそ {name} さん！')

# # フォーム
# with st.form(key='profile_form'):
#     # テキストボックス
#     name = st.text_input('名前')
#     company = st.text_input('会社名')

#     # セレクトボックス
#     age_category = st.selectbox('年齢層', ('18歳未満', '18～25歳', '25～30歳', '31～40歳', '41～50歳', '51～60歳', '61歳以上'))

#     # スライダー
#     height = st.slider('身長', min_value=120, max_value=210)

#     # 日付
#     start_date = st.date_input('開始日', 'today')

#     submit_btn = st.form_submit_button('Submit')
#     cancel_btn = st.form_submit_button('Cancel')

#     if submit_btn:
#         st.text(f'ようこそ {company} の {name} さん！')
#         st.text(f'年齢層：{age_category}')
#         st.text(f'身長：{height}')
#         st.text(f'開始日：{start_date}')
