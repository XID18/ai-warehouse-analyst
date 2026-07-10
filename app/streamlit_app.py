import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Warehouse Analyst", page_icon="📦")
st.title("📦 AI Warehouse Analyst")

tab_chat, tab_forecast, tab_abcxyz = st.tabs(["Чат", "Прогноз по SKU", "ABC-XYZ"])

with tab_chat:
    st.subheader("Спросить про склад")
    question = st.text_input("Например: какие аналоги есть у артикула X?")
    if st.button("Спросить") and question:
        with st.spinner("Думаю..."):
            resp = httpx.post(f"{API_URL}/chat", json={"question": question}, timeout=60)
            st.write(resp.json()["answer"])

with tab_forecast:
    st.subheader("Прогноз спроса по SKU")
    sku = st.text_input("Артикул SKU")
    if st.button("Спрогнозировать") and sku:
        resp = httpx.post(f"{API_URL}/forecast", json={"sku": sku}, timeout=30)
        st.metric("Прогноз на следующий период", resp.json()["forecast"])

with tab_abcxyz:
    st.subheader("ABC-XYZ классификация каталога")
    if st.button("Загрузить классификацию"):
        resp = httpx.get(f"{API_URL}/abc-xyz", timeout=30)
        st.dataframe(resp.json())
