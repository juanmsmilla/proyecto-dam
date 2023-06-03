from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
from backend.core import index_pdf_FAISS, generate_response
from static.styles import HIDE_ST_STYLE
from currency_converter import CurrencyConverter



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

if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0

if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = 0



# índice vectorial texto del pdf. Usa FAISS y OpenAI embeddings
with st.sidebar:

    # define 2 columnas iguales dentro del sidebar
    col1, col2 = st.columns([1, 1])
    USD = st.session_state["total_cost"]
    EUR = CurrencyConverter().convert(USD, 'USD', 'EUR')
    col1.markdown("### Coste")
    col1.markdown(f"# {EUR: .2f} €")
    tokens = st.session_state["total_tokens"]
    col2.markdown("### Tokens")
    col2.markdown(f"# {tokens}")


    with get_openai_callback() as cb:
        pdf_vector_index = index_pdf_FAISS()
        st.session_state["total_cost"] += cb.total_cost
        st.session_state["total_tokens"] += cb.total_tokens


if pdf_vector_index is not None:

    # user input
    user_input = st.text_input("Pregunta a tu PDF:")


    if user_input:
        # spinner es un widget UI.
        with st.spinner("Generando respuesta..."):
            with get_openai_callback() as cb:
                generated_response = generate_response(
                    pdf_vector_index,
                    query=user_input,
                    chat_history=st.session_state["chat_history"]
                    )

                # añadir datos a la sesión de streamlit
                st.session_state["user_prompt_history"].append(user_input)
                st.session_state["chat_answer_history"].append(generated_response["answer"])
                st.session_state["chat_history"].append((user_input, generated_response["answer"]))

                # añadir costo
                st.session_state["total_cost"] += cb.total_cost
                st.session_state["total_tokens"] += cb.total_tokens

    if st.session_state["chat_answer_history"]:
        for generated_response, user_query in zip(
                st.session_state["chat_answer_history"],
                st.session_state["user_prompt_history"]
        ):
            message(user_query, is_user=True)
            message(generated_response)

cost = st.session_state["total_cost"]
st.write(f"Coste acumulado de la sesión {cost}")



