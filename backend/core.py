from typing import Any, List, Tuple
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader




# el decorador cachea el return. necesario para no ser borrado al reinicio de sesion
@st.cache_data(experimental_allow_widgets=True)
def index_pdf_FAISS(chunk_size: int = 1000, chunk_overlap: int = 200):

    # widget
    pdf = st.file_uploader("Arrastra aquí tu PDF", type="pdf")
    # extraer texto
    if pdf is not None:
        pdf_reader = PdfReader(pdf, )
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

        # FAISS
        embeddings = OpenAIEmbeddings()
        vector_index = FAISS.from_texts(chunks, embeddings)

        return vector_index



def generate_response(
        vector_index: Any,
        query: str,
        chat_history: List[Tuple[str, Any]] = [],
) -> Any:
    # verbose=True - respuestas menos concisas
    # temperature - valores del 0(min) al 1(max)
    formated_query = f'Busca la siguiente pregunta en el documento: `{query}`. Si no encuentas la respuesta' \
                     f'responde unicamente: "El documento no contiene información relacionada."'
    chat = ChatOpenAI(verbose=True, temperature=0.2)

    answer = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vector_index.as_retriever()
    )

    return answer({"question": formated_query, "chat_history": chat_history})


# def generate_response_context(input_text, faiss_index):
#     memory = st.session_state["memory"]
#     memory.save_context({"input": input_text}, {})
#     conversation_history = memory.load_memory_variables({})['history']
#     # pasajes relevantes
#     docs = faiss_index.similarity_search()
#     response = llm.generate_text(prompt=conversation_history + docs)
#
#     memory.save_context({}, {"output": response})
#
#     response
