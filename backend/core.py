from typing import Any, List, Tuple
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader


# @st.cache_data(experimental_allow_widgets=True)
def get_pdf_text():
    pdf = st.file_uploader("Arrastra aquí tu PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf, )
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text


def add_pdfs(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf, )
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


# el decorador cachea el return. necesario para no ser borrado al reinicio de sesion
# @st.cache_data(experimental_allow_widgets=True)
def index_pdf_faiss(pdf_text, chunk_size: int = 1000, chunk_overlap: int = 200):

    # chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(pdf_text)

    # FAISS
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_texts(chunks, embeddings)

    return vector_index


def generate_response(
        vector_index: Any,
        query: str,
        chat_history: List[Tuple[str, Any]] = []):
    # buscar info solo en pdf


    formated_query = f'Responde a la pregunta `{query}` limitando el contexto a los documentos proporcionados. ' \
                     f'Si no encuentras información sobre la pregunta en el contexto proporcionado responde:' \
                     f'"El contexto no contiene información relevante a la pregunta.'


    # verbose=True - respuestas menos concisas
    # temperature - valores del 0(min) al 1(max)
    llm = ChatOpenAI(verbose=True, temperature=0.2)

    answer = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_index.as_retriever()
    )

    return answer({"question": formated_query, "chat_history": chat_history})


def load_qa_memory(query, answer) -> None:
    st.session_state["user_prompt_history"].append(query)
    st.session_state["chat_answer_history"].append(answer["answer"])
    st.session_state["chat_history"].append((query, answer["answer"]))


def load_costs_memory(cb) -> None:
    st.session_state["total_cost"] += cb.total_cost
    st.session_state["total_tokens"] += cb.total_tokens


def initialize_session_var(name: str, val: Any) -> None:
    if name not in st.session_state:
        st.session_state[name] = val
