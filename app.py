"""
app.py
Simple Streamlit interface for the RAG chatbot.
Displays top 3 retrieved chunks with similarity scores and the LLM answer.
"""
import streamlit as st
from rag_chatbot import retrieve_multi, generate_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG Chatbot - Bakery Ingredients")
st.markdown("Enter a query to retrieve the most relevant chunks and get an LLM-generated answer.")

query = st.text_input("Your question")

if st.button("Search") and query.strip():
    with st.spinner("Retrieving chunks and generating answer..."):
        passages = retrieve_multi(query)
        answer = generate_answer(query, passages)

    st.subheader("Top Retrieved Chunks")

    for i, p in enumerate(passages, 1):
        score = p.get("rerank_score", p["score"])
        similarity = p["score"]
        ingredient = p["metadata"].get("ingredient", "N/A")

        st.markdown(f"**Chunk {i}**")
        st.markdown(f"- **Ingredient:** {ingredient}")
        st.markdown(f"- **Cosine Similarity:** {similarity}")
        if "rerank_score" in p:
            st.markdown(f"- **Rerank Score:** {round(p['rerank_score'], 4)}")
        st.text_area(
            f"Text (chunk {i})",
            value=p["text"],
            height=120,
            key=f"chunk_{i}",
            disabled=True,
        )
        st.divider()

    st.subheader("LLM Answer")
    st.write(answer)
