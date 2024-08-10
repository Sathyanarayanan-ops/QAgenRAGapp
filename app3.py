import logging
import random
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ["GROQ_API_KEY"] = ""

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def load_and_split_document(pdf_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logging.info(f"Loaded {len(pages)} pages from PDF")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)
    logging.info(f"Split documents into {len(docs)} chunks")
    return docs

def setup_llm_and_embeddings(model_name):
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    embeddings = HuggingFaceEmbeddings()
    return llm, embeddings

def create_vector_store(docs, embeddings):
    return Chroma.from_documents(documents=docs, embedding=embeddings)

def generate_questions(llm, context, num_questions):
    question_gen_template = """
    Based on the following text, generate {number_questions} multiple-choice questions. 
    For each question:
    1. Provide the question
    2. Provide exactly four options labeled A, B, C, and D
    3. Indicate the correct answer

    Use this format for each question:
    Q: [Question text]
    A: [Option A]
    B: [Option B]
    C: [Option C]
    D: [Option D]
    Correct Answer: [A/B/C/D]

    Text:
    {context}

    Generate {number_questions} multiple-choice questions:
    """
    
    question_gen_prompt = PromptTemplate(template=question_gen_template, input_variables=["context", "number_questions"])
    question_gen_chain = LLMChain(llm=llm, prompt=question_gen_prompt)

    response = question_gen_chain.run(context=context, number_questions=num_questions)
    questions = response.split("\n\n")
    questions = [q.strip() for q in questions if q.strip() and q.startswith("Q:")]
    logging.info(f"Generated {len(questions)} questions")
    return questions

def generate_answers(llm, questions, vectorstore, context):
    qa_template = """
    Based on the following context and the given multiple-choice question, explain why the indicated correct answer is right.
    If the correct answer seems wrong based on the context, explain why and suggest what the correct answer should be.

    Context:
    {context}

    Question and Options:
    {question}

    Provide an explanation for the correct answer:
    """

    qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    qa_pairs = []
    for i, question in enumerate(questions, 1):
        try:
            relevant_docs = vectorstore.similarity_search(question, k=2)
            local_context = "\n".join([doc.page_content for doc in relevant_docs])
            explanation = qa_chain.run(context=local_context, question=question)
            qa_pairs.append((question, explanation))
            logging.info(f"Generated answer for question {i}")
        except Exception as e:
            logging.error(f"Error generating answer for question {i}: {e}")
    return qa_pairs

def main():
    st.title("MCQ Generator")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)

    if st.button("Generate MCQs"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF and generating questions..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    pdf_path = tmpfile.name 

                setup_logging()
                docs = load_and_split_document(pdf_path, 1000, 200)
                llm, embeddings = setup_llm_and_embeddings("llama3-8b-8192")
                vectorstore = create_vector_store(docs, embeddings)

                sample_size = min(20, len(docs))
                sampled_docs = random.sample(docs, sample_size)
                context = "\n".join([doc.page_content for doc in sampled_docs])
                
                questions = generate_questions(llm, context, num_questions)
                qa_pairs = generate_answers(llm, questions, vectorstore, context)

                for i, (question, explanation) in enumerate(qa_pairs, 1):
                    st.subheader(f"Question {i}")
                    lines = question.split("\n")
                    st.write(lines[0])  # Question text
                    for option in lines[1:5]:
                        st.write(option)  # Options A, B, C, D
                    st.write(lines[5] if len(lines) > 5 else "Correct Answer: Not specified")  # Correct Answer
                    st.write("Explanation:")
                    st.write(explanation)
                    st.write("---")

                os.unlink(pdf_path)
        else:
            st.error("Please upload a PDF file")

if __name__ == "__main__":
    main()
