from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from backend.core import index_pdf_FAISS, generate_response
from static.styles import HIDE_ST_STYLE


load_dotenv()


# st.set_page_config(page_title="PDF bot")
st.header("PDF chatbot - Juan Miguel Sánchez Milla")
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

# todo next line not needed(?)
# docs = vector_index.similarity_search(query)


if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []



# pdf = st.sidebar.file_uploader("Arrastra un .pdf", type="pdf", accept_multiple_files=True)

# índice vectorial del texto del pdf. Usa FAISS y OpenAI embeddings

pdf_vector_index = index_pdf_FAISS()

if pdf_vector_index is not None:

    # user input
    user_input = st.text_input("Pregunta a tu PDF:")


    if user_input:
        # spinner es un widget UI.
        with st.spinner("Generando respuesta..."):
            generated_response = generate_response(
                pdf_vector_index,
                query=user_input,
                chat_history=st.session_state["chat_history"]
            )

            # añadir datos a la sesión de streamlit
            st.session_state["user_prompt_history"].append(user_input)
            st.session_state["chat_answer_history"].append(generated_response["answer"])
            st.session_state["chat_history"].append((user_input, generated_response["answer"]))


    if st.session_state["chat_answer_history"]:
        for generated_response, user_query in zip(
                st.session_state["chat_answer_history"],
                st.session_state["user_prompt_history"]
        ):
            message(user_query, is_user=True)
            message(generated_response)




