import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="Write your API Key here")



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config(page_title="📖 Talking PDF", page_icon="🤖", layout="wide")

    # Header
    st.markdown(
        """
        <div style="text-align:center; padding:20px;">
            <h1 style="color:#4CAF50;">🤖 Talk with Your PDF</h1>
            <p style="font-size:18px;">Upload PDF files and ask questions powered by <b>Google Gemini</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload & Process")
        pdf_docs = st.file_uploader(
            "Upload your PDF files", accept_multiple_files=True, type=["pdf"]
        )
        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDF processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

    # Question Box
    st.markdown("### 💬 Ask a Question")
    user_question = st.text_input("Type your question here and press Enter...")
    if user_question:
        with st.spinner("Thinking... 🤔"):
            answer = user_input(user_question)

        st.markdown(
            f"""
            <div style="background-color:#f1f3f6; padding:20px; border-radius:10px; margin-top:20px;">
                <h4 style="color:#333;">📌 Answer:</h4>
                <p style="font-size:16px; color:#000;">{answer}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
        <hr>
        <div style="text-align:center; font-size:14px; color:gray;">
            Made with ❤️ using Streamlit & LangChain | Powered by Google Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )



if __name__ == "__main__":
    main()