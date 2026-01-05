import streamlit as st
import requests


st.title("RAG Assistant")
if st.button("Check backend"):
    r = requests.get("http://0.0.0.0:8000/health")
    st.json(r.json())