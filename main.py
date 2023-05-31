from dotenv import load_dotenv
import streamlit as st
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback
from streamlit_chat import message

from backend.core import indexing_pdf, generate_answer


load_dotenv()

def main():
    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []

    if "chat_answer_history" not in st.session_state:
        st.session_state["chat_answer_history"] = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.set_page_config(page_title="PDF bot")
    st.header("PDF chatbot - Juan Miguel Sánchez Milla")

    if "pdf" not in st.session_state:
        pdf = st.file_uploader("Adjuntar pdf", type="pdf")
        st.session_state["pdf"] = pdf

    if pdf is not None:
        # índice vectorial del texto del pdf. Usa FAISS
        vector_index = indexing_pdf(pdf)

        # user input
        user_input = st.text_input("Pregunta a tu PDF:")

        if st.session_state["chat_answer_history"]:
            for generated_response, user_query in zip(
                st.session_state["chat_answer_history"],
                st.session_state["user_prompt_history"]
            ):
                message(user_query, is_user=True)
                message(generated_response)

        if user_input:
            with st.spinner("Generando respuesta..."):
                generated_response = generate_answer(
                    vector_index,
                    query=user_input,
                    chat_history=st.session_state["chat_history"]
                )

                # añadir datos a la sesión de streamlit
                st.session_state["user_prompt_history"].append(user_input)
                st.session_state["chat_answer_history"].append(generated_response)
                st.session_state["chat_history"].append((user_input, generated_response["answer"]))
        # if user_question:
            # documentos(langchain) similares a user_question
            # docs = vector_index.similarity_search(user_question)

            # llm = OpenAI()
            # chain = load_qa_chain(llm, chain_type="stuff")
            # with get_openai_callback() as cb:
            #     response = chain.run(input_documents=docs, question=user_question)
            #     print(cb)
            #
            # st.write(response)



if __name__ == '__main__':
    main()
