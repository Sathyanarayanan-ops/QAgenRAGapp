
# MCQ Generator App

Welcome to the MCQ Generator App! This application generates high-quality multiple-choice questions (MCQs) from PDF documents using advanced AI models. It provides a streamlined way to create educational content, ideal for instructors, students, and anyone interested in automated question generation.

## Features

- **PDF Document Loading**: Easily upload PDF files to extract content.
- **Text Splitting**: Documents are split into manageable chunks using recursive character text splitting.
- **LLM-Powered Question Generation**: Utilizes the `ChatGroq` model to generate well-structured MCQs.
- **Answer Generation**: Leverages vector stores and embeddings for accurate answer predictions.
- **Streamlit Interface**: User-friendly UI to upload PDFs, specify question count, and view generated MCQs.
<img width="1041" alt="Screenshot 2024-08-12 at 10 49 22 AM" src="https://github.com/user-attachments/assets/963854f4-e3fd-4689-8f26-1e67e161a563">
<img width="995" alt="Screenshot 2024-08-12 at 4 11 50 PM" src="https://github.com/user-attachments/assets/9edf88ff-d53d-4aea-876a-29c6fd79f39e">

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/mcq-generator-app.git
cd mcq-generator-app
pip install -r requirements.txt
```

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
2. **Upload PDF**: Choose a PDF file from your local machine.
3. **Set Question Count**: Specify the number of MCQs to generate.
4. **Generate Questions**: Click the button and let the app process the document.
5. **View Results**: Generated questions and answers are displayed on the interface.

## Configuration

- **Model Setup**: Ensure your environment is configured with the `GROQ_API_KEY`.
- **Document Chunking**: Adjust `chunk_size` and `chunk_overlap` in the `load_and_split_document` function to optimize text splitting.

## Logging

Logs are generated to track the process of document loading, question generation, and answer generation.

## Contributing

Feel free to fork the repository, submit pull requests, and contribute to improving the application.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://www.streamlit.io/)
- [Hugging Face](https://huggingface.co/)

---

