import streamlit as st
import requests

st.title("RAG Ассистент")

if st.button("Доступ к сервису"):
    try:
        r = requests.get("http://127.0.0.1:8000/health")
        if r.status_code == 200:
            st.success(r.json())
        else:
            st.error(f"Ошибка сервиса: {r.status_code}")
    except Exception as e:
        st.error(f"Ошибка запроса: {e}")

operator = st.selectbox("Выберите роль ассистента", ["ERP", "HR", "Finance"])
language = st.selectbox("Язык", ["ru", "en"])
query = st.text_input("Ваш вопрос:")

if st.button("Отправить") and query:
    payload = {
        "query": query,
        "mode": operator,
        "language": language
    }
    try:
        r = requests.post("http://127.0.0.1:8000/chat", json=payload)
        data = r.json()

        st.markdown(f"Вы: {query}")
        st.markdown(f"Бот: {data['answer']}")

    except Exception as e:
        st.error(f"Ошибка запроса: {e}")
