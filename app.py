

#######################################################################


import logging
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os 
import streamlit as st 
import tempfile 
from langchain_community.embeddings import HuggingFaceEmbeddings


os.environ["GROQ_API_KEY"] = "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"

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
    #llm = Ollama(model=model_name)
    llm = ChatGroq(
        #groq_api_key = "gsk_svavVx9y9Zq5BvmxKVGEWGdyb3FYQWmB1x2MTEaZRsTXo8cNzB2",
        model = model_name,
        temperature=0,
        max_tokens = None,
        timeout = None,
        max_retries=2
    )

    #embeddings = OllamaEmbeddings(model=model_name, show_progress=True)
    # hugging face inference embeddings check
    # problem in prompting , be more specific 
    # Try mcq format 
    # Next work on text to sql 

    embeddings = HuggingFaceEmbeddings()
    return llm, embeddings

def create_vector_store(docs, embeddings):
    return Chroma.from_documents(documents=docs, embedding=embeddings)

def generate_questions(llm, context, num_questions):
    question_gen_template = """
    Based on the following text from a book, generate exactly {number_questions} diverse and insightful questions. 
    Focus on the core concepts and main ideas of the book. Ignore any metadata like table of contents, copyright information, publication year, or author details.
    Ask questions that would help a reader understand the key points and arguments of the book.
    Text:
    {context}

    Generate{number_questions} questions:

    """
    # messages = [
    #     ("system", "You are a helpful assistant that generates questions based on given context."),
    #     ("human", question_gen_template.format(context=context, number_questions=num_questions))
    # ]
    # response = llm.invoke(messages)
    # questions = response.content.split("\n")
    # questions = [q.strip() for q in questions if q.strip()]
    # logging.info(f"Generated {len(questions)} questions")
    # return questions

    question_gen_prompt = PromptTemplate(template=question_gen_template, input_variables=["context", "number_questions"])
    question_gen_chain = LLMChain(llm=llm, prompt=question_gen_prompt)

    questions = question_gen_chain.run(context=context, number_questions=num_questions).split("\n")
    questions = [q.strip() for q in questions if q.strip() and q.strip]  # Remove empty questions
    logging.info(f"Generated {len(questions)} questions")
    return questions

def generate_answers(llm, questions, vectorstore, context):
    qa_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer:"""

    # qa_pairs = []
    # for i, question in enumerate(questions, 1):
    #     try:
    #         relevant_docs = vectorstore.similarity_search(question, k=2)
    #         context = "\n".join([doc.page_content for doc in relevant_docs])
    #         messages = [
    #             ("system", "You are a helpful assistant that answers questions based on given context."),
    #             ("human", qa_template.format(context=context, question=question))
    #         ]
    #         response = llm.invoke(messages)
    #         answer = response.content
    #         qa_pairs.append((question, answer))
    #         logging.info(f"Generated answer for question {i}")
    #     except Exception as e:
    #         logging.error(f"Error generating answer for question {i}: {e}")
    # return qa_pairs




    qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    qa_pairs = []
    for i, question in enumerate(questions, 1):
        try:
            relevant_docs = vectorstore.similarity_search(question, k=2)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            answer = qa_chain.run(context=context, question=question)
            qa_pairs.append((question, answer))
            logging.info(f"Generated answer for question {i}")
        except Exception as e:
            logging.error(f"Error generating answer for question {i}: {e}")
    return qa_pairs
import random 
def main():
    st.title("QA generator")
    uploaded_file = st.file_uploader("Choose a PDF file",type = "pdf")
    num_questions = st.number_input("NUmber of questions", min_value = 1, max_value = 20 )

    if st.button("Generate"):
        if uploaded_file is not None :
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmpfile:
               tmpfile.write(uploaded_file.getvalue())
               pdf_path = tmpfile.name 

            setup_logging()
            #pdf_path = "/Users/sathya/Desktop/Rag/ECE5554 SU24 Computer Vision Syllabus.pdf"
            docs = load_and_split_document(pdf_path,1000,200)
            llm, embeddings = setup_llm_and_embeddings("llama3-8b-8192")
            vectorstore = create_vector_store(docs, embeddings)

            sample_size = min(20,len(docs))
            sampled_docs = random.sample(docs,sample_size)
            context = "\n".join([doc.page_content for doc in sampled_docs])  # Use first 5 chunks ----> changed to sample up to 20 chunks 
            
            questions = generate_questions(llm, context, num_questions)
            qa_pairs = generate_answers(llm, questions, vectorstore, context)

    # Print results
            for i, (question, answer) in enumerate(qa_pairs, 1):
                st.subheader(f"Q{i}: {question}")
                st.write(f"A{i}: {answer}\n")

            os.unlink(pdf_path)
        else:
            st.error("PLease upload a PDF file")
    


if __name__ == "__main__":
    main()