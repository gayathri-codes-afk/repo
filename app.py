import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DocuChat RAG - SMI Interview", page_icon="📄")
st.title("📄 DocuChat - RAG Based Document Q&A")
st.markdown("**For SMI Techsolutions - Junior AI Engineer Interview | 9th Oct 2026**")

def get_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "
    return text

def get_chunks(text):
    chunk_size = 600
    overlap = 100
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def find_answer(user_question, text_chunks):
    vectorizer = TfidfVectorizer().fit(text_chunks + [user_question])
    vectors = vectorizer.transform(text_chunks + [user_question])
    question_vector = vectors[-1]
    chunk_vectors = vectors[:-1]
    scores = cosine_similarity(question_vector, chunk_vectors)
    best_index = scores.argmax()
    best_score = scores[0][best_index]
    return text_chunks[best_index], float(best_score)

st.sidebar.header("How RAG Works")
st.sidebar.write("1. PDF Upload -> Read")
st.sidebar.write("2. Chunking (600 chars)")
st.sidebar.write("3. TF-IDF Embedding")
st.sidebar.write("4. Cosine Similarity Retrieval")
st.sidebar.write("5. Answer Generation")

pdf_file = st.file_uploader("Upload PDF", type=['pdf'])

if pdf_file is not None:
    with st.spinner("Processing PDF..."):
        raw_text = get_pdf_text(pdf_file)
        all_chunks = get_chunks(raw_text)
    st.success(f"Done! Created {len(all_chunks)} chunks from PDF")
    with st.expander("See first chunk"):
        st.write(all_chunks[0][:500])
    user_q = st.text_input("Ask question from your PDF:")
    if user_q:
        answer, score = find_answer(user_q, all_chunks)
        st.subheader("Retrieved Answer:")
        st.write(answer)
        st.caption(f"Similarity Score: {score:.2f}")
else:
    st.info("Upload any PDF to start - Try with your RESUME")