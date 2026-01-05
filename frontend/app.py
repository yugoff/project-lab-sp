import streamlit as st
import requests


st.title("RAG Ассистент")

if "history" not in st.session_state:
    st.session_state.history = []

operator = st.selectbox("Выберите роль ассистента", ["ERP", "HR", "Finance"])
language = st.selectbox("Язык", ["ru", "en"])

query = st.text_input("Ваш вопрос:")

if st.button("Отправить") and query:
    payload = {"query": query, "mode": operator, "language": language}
    try:
        r = requests.post("http://0.0.0.0:8000/chat", json=payload)
        data = r.json()
        st.session_state.history.append(("Вы", query))
        st.session_state.history.append(("Бот", data["answer"]))
    except Exception as e:
        st.error(f"Ошибка запроса: {e}")

for speaker, text in st.session_state.history:
    if speaker == "Вы":
        st.markdown(f"**{speaker}:** {text}")
    else:
        st.markdown(f"**{speaker}:** {text}")
