from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader


def indexing_pdf(pdf, chunk_size: int = 1000, chunk_overlap: int = 200):
    # extraer texto
    pdf_reader = PdfReader(pdf, )
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
        )
    chunks = text_splitter.split_text(text)

    # FAISS
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_texts(chunks, embeddings)

    return vector_index
