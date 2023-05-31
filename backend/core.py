from typing import Any, List, Tuple
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader


# el decorador cachea el archivo

# @st.experimental_memo(experimental_allow_widgets=True)
@st.cache_data(experimental_allow_widgets=True)
def get_pdf():
    return st.file_uploader("Arrastra aquí tu PDF", type="pdf")


def indexing_pdf(pdf, chunk_size: int = 1000, chunk_overlap: int = 200):
    # extraer texto
    st.write("indexing_pdf")
    pdf_reader = PdfReader(pdf, )
    st.write(type(pdf))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
        )
    chunks = splitter.split_text(text)
    st.write("chunk echos")

    # FAISS
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_texts(chunks, embeddings)

    return vector_index


def generate_answer(
        vector_index: Any,
        query: str,
        chat_history: List[Tuple[str, Any]] = [],
) -> Any:
    st.write("generate_answer")
    # verbose=True - respuestas menos concisas
    # temperature=0 - respuestas no creativas
    chat = ChatOpenAI(verbose=True, temperature=0)

    answer = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vector_index.as_retriever()
    )
    st.write(f"answer is {answer}")

    return answer({"question": query, "chat_history": chat_history})
