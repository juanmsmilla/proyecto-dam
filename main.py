from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
from backend.core import get_pdf_text, index_pdf_faiss, generate_response, load_qa_memory, load_costs_memory, initialize_session_var
from static.styles import HIDE_ST_STYLE
from currency_converter import CurrencyConverter

load_dotenv()

# st.set_page_config(page_title="PDF bot")
st.header("PDF chatbot - Juan Miguel Sánchez Milla")
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

initialize_session_var("user_prompt_history", [])
initialize_session_var("chat_answer_history", [])

initialize_session_var("chat_history", [])
initialize_session_var("total_cost", 0.0)
initialize_session_var("total_tokens", 0)
# initialize_session_var("pdf_text", get_pdf_text())

if "pdf_text" in st.session_state:
    # define 2 columnas iguales dentro del sidebar
     col1, col2 = st.columns([1, 1])
     usd = st.session_state["total_cost"]
     # EUR = CurrencyConverter().convert(USD, 'USD', 'EUR')
     col1.markdown("### Coste")
     col1.markdown(f"# {usd} $")
     tokens = st.session_state["total_tokens"]
     col2.markdown("### Tokens")
     col2.markdown(f"# {tokens}")


if "pdf_text" not in st.session_state:
    pdf_text = get_pdf_text()
    if pdf_text:
        st.session_state["pdf_text"] = pdf_text

# pdf_text = get_pdf_text()
# if "user_prompt_history" not in st.session_state:
#     st.session_state["user_prompt_history"] = []
#
# if "chat_answer_history" not in st.session_state:
#     st.session_state["chat_answer_history"] = []
#
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
#
# if "total_cost" not in st.session_state:
#     st.session_state["total_cost"] = 0.0
#
# if "total_tokens" not in st.session_state:
#     st.session_state["total_tokens"] = 0


# índice vectorial texto del pdf. Usa FAISS y OpenAI embeddings
# with st.sidebar:
#     # define 2 columnas iguales dentro del sidebar
#     col1, col2 = st.columns([1, 1])
#     USD = st.session_state["total_cost"]
#     EUR = CurrencyConverter().convert(USD, 'USD', 'EUR')
#     col1.markdown("### Coste")
#     col1.markdown(f"# {EUR: .2f} €")
#     tokens = st.session_state["total_tokens"]
#     col2.markdown("### Tokens")
#     col2.markdown(f"# {tokens}")

    # st.session_state["total_cost"] += cb.total_cost
    # st.session_state["total_tokens"] += cb.total_tokens


if "pdf_text" in st.session_state:
    with get_openai_callback() as cb:
        pdf_vector_index = index_pdf_faiss(st.session_state["pdf_text"])
        if pdf_vector_index is not None:
            user_input = st.text_input("Hazme una pregunta 🤖")
            if user_input:
                # spinner es un widget UI.
                with st.spinner("Generando respuesta..."):

                    generated_response = generate_response(
                        pdf_vector_index,
                        query=user_input,
                        chat_history=st.session_state["chat_history"]
                    )

                    load_qa_memory(user_input, generated_response)
                        # # añadir datos a la sesión de streamlit
                        # st.session_state["user_prompt_history"].append(user_input)
                        # st.session_state["chat_answer_history"].append(generated_response["answer"])
                        # st.session_state["chat_history"].append((user_input, generated_response["answer"]))
                        #
                        # # añadir costo
                        # st.session_state["total_cost"] += cb.total_cost
                        # st.session_state["total_tokens"] += cb.total_tokens

                if st.session_state["chat_answer_history"]:
                    for generated_response, user_query in zip(
                            st.session_state["chat_answer_history"],
                            st.session_state["user_prompt_history"]
                    ):
                        message(user_query, is_user=True)
                        message(generated_response)
    load_costs_memory(cb)
