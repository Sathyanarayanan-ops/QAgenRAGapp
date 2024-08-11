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


os.environ["GROQ_API_KEY"] = "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"


def setup_logging():
    logging.basicConfig(level=logging.INFO)

def load_and_split_document(pdf_path, chunk_size , chunk_overlap):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logging.info(f"Loaded {len(pages)} pages from PDF")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    #text_splitter = SemanticChunker(embeddings)
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
    #hugging face inference
    embeddings = HuggingFaceEmbeddings()
    return llm, embeddings

def create_vector_store(docs, embeddings):
    return Chroma.from_documents(documents=docs, embedding=embeddings)

#Role is missing in prompt
#granular level prompt 
# embeddings hugging face  inference
# no expln needed 
# use 70 b 
# vector db not needed -- research
# do more research on inference 
# semantic chunking 

def generate_questions(llm, context, num_questions):
    question_gen_template = """
    You are an expert in generating educational content and question generation. Your task is to create {number_questions} high-quality multiple-choice questions
    based on the following text

    For each question, adhere by the following instructions very carefully and strictly:
    1. **Read the provided text carefully.**
    2. **Identify key concepts or facts from the text that can be turned into a question.**
    3. **Formulate a clear and concise question based on these key concepts or facts, make sure to understand that there are a lot of metadata and unwanted things such as details about the author name, or the publisher, page number,
      date of publication, etc just to name a few; you should realise that these are things that are not essential facts about the topic, this is a very strict rule.**
    4. **Generate exactly four distinct and plausible options labeled A, B, C and D. Ensure that only one option is correct and the other three are relevant information but are not the correct answer; remember only one option out of the four is correct. **
    5. **When you generate a question, make sure that there is sufficient information, do not ask questions such as "What does section 5.1 talk about", the goal of the questions is to test the concepts and knowledge of the user, not to make them memorize what every section does. **
    6. ** If you do decide to ask question about a particular section say section 2.1, give enough information regarding the section so that the question makes sense to the user, if you are talking about a particular equation, say equation 3.3, list out the equation so that the user knows what you are talking about.**

    Use this format for each question`:
    Q: [Question text]`
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
    
    all_questions = [q.strip() for q in response.split("Q:") if q.strip()]
    
    valid_questions = []
    for q in all_questions:
        lines = q.split("\n")
        if len(lines) >= 6 and all(option in q for option in ['A:', 'B:', 'C:', 'D:']) and 'Correct Answer:' in q:
            valid_questions.append(q)
        else:
            logging.warning(f"Skipping improperly formatted question: {q}")
    
    logging.info(f"Generated a total of {len(valid_questions)} valid questions")
    return valid_questions[:num_questions]

def generate_answers(llm, questions, vectorstore, context):
    qa_template = """
    You are an expert in answering multiple choice questions, based on the following context and the given multiple-choice question, choose the right option; make sure to analyze all the facts and core topics of the subject before answering the question.
    You shall choose only one option, which is the correct answer for the question. No need to explain why you chose a particular answer, just choosing the option is enough.

    Context:
    {context}

    Question and Options:
    {question}

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
    num_questions = st.number_input("Number of questions", min_value=5, max_value=100, value=5) # min 5 and max 50 

    if st.button("Generate MCQs"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF and generating questions..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    pdf_path = tmpfile.name 

                setup_logging()
                llm, embeddings = setup_llm_and_embeddings("llama3-70b-8192") # use 70b next
                docs = load_and_split_document(pdf_path, 1000, 200)
                # docs = load_and_split_document(pdf_path, embeddings)

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
                    st.write(lines[5]) #if len(lines) > 7 else "Correct Answer: Not specified")  # Correct Answer
                    st.write("Correct Option:")
                    #st.write(explanation)
                    st.write("---")

                os.unlink(pdf_path)
        else:
            st.error("Please upload a PDF file")

if __name__ == "__main__":
    main()

