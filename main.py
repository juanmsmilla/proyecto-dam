from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from backend.core import indexing_pdf


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF bot")
    st.header("PDF chatbot - Juan Miguel Sánchez Milla")

    pdf = st.file_uploader("Adjuntar pdf", type="pdf")

    if pdf is not None:
        # índice vectorial del texto del pdf. Usa FAISS
        vector_index = indexing_pdf(pdf)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # documentos(langchain) similares a user_question
            docs = vector_index.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
