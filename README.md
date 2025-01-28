# Chatbot with Contextual Responses from Website Scrapes

This project is a chatbot capable of answering questions based on the content scraped from websites. It leverages various tools and frameworks to efficiently scrape, process, and store data, enabling contextual responses via a user-friendly UI built with Streamlit.

## Features
- Scrapes and processes website content to extract contextual information.
- Stores processed data in a vector store for efficient retrieval.
- Utilizes advanced language models for accurate and contextual question answering.
- Provides an interactive user interface for seamless interactions.

---

## Frameworks and Tools Used
1. **Embedding Model**: [HuggingFaceInstructEmbeddings](https://huggingface.co/) from `langchain_community` - Generates embeddings for text.
2. **Scraping Tool**: [WebBaseLoader](https://langchain.readthedocs.io/en/latest/modules/document_loaders/examples/web_base.html) from `langchain_community` - Scrapes website content.
3. **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) from `langchain_community` - Stores text embeddings for efficient similarity search.
4. **QA Chain**: `create_stuff_documents_chain` from `langchain_community` - Handles question answering based on retrieved information.
5. **Prompt Template**: `ChatPromptTemplate` from `langchain_community` - Builds and handles custom prompts.
6. **Retrieval Chain**: `create_retrieval_chain` from `langchain_community` - Retrieves relevant data from the vector store.
7. **Language Model**: `ChatGoogleGenerativeAI` from `langchain_community` - Implements "gemini-pro" LLM for generating responses.
8. **UI Framework**: [Streamlit](https://streamlit.io/) - Builds the chatbot's interactive interface.

---

## How It Works

### 1. **Scraping Website Content**
The `WebBaseLoader` module from `langchain_community` is used to scrape data from target websites. It extracts and preprocesses the website's textual content, preparing it for further processing.

**Code Snippet:**
```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()
```

### 2. **Generating Text Embeddings**
Using `HuggingFaceInstructEmbeddings`, the text from the scraped documents is converted into vector embeddings. These embeddings represent the semantic meaning of the text, enabling efficient similarity searches.

**Code Snippet:**
```python
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

embedding_model = HuggingFaceInstructEmbeddings()
embeddings = embedding_model.embed_documents([doc.text for doc in documents])
```

### 3. **Storing Embeddings in a Vector Store**
The `FAISS` vector store is utilized to store and manage the generated embeddings. It enables fast retrieval of similar text segments during the question-answering process.

**Code Snippet:**
```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_embeddings(embeddings, documents)
```

### 4. **Building the QA Chain**
The `create_stuff_documents_chain` function from `langchain_community` is used to implement a QA pipeline. This chain processes the retrieved text and generates relevant answers.

**Code Snippet:**
```python
from langchain_community.chains import create_stuff_documents_chain

qa_chain = create_stuff_documents_chain()
```

### 5. **Handling Prompts**
`ChatPromptTemplate` is used to create and manage the prompts sent to the language model. It ensures the prompts are formatted correctly for the model to understand the context.

**Code Snippet:**
```python
from langchain_community.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate("Provide an answer based on the context:")
```

### 6. **Retrieving Relevant Data**
The `create_retrieval_chain` function connects the vector store with the QA chain. It retrieves relevant text segments from the vector store based on user queries.

**Code Snippet:**
```python
from langchain_community.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(vector_store, qa_chain)
```

### 7. **Generating Answers with "gemini-pro" LLM**
The `ChatGoogleGenerativeAI` module is used to utilize the "gemini-pro" language model. It generates natural and accurate answers based on the retrieved content.

**Code Snippet:**
```python
from langchain_community.llms import ChatGoogleGenerativeAI

generative_ai = ChatGoogleGenerativeAI(model_name="gemini-pro")
```

### 8. **Building the User Interface**
The chatbot's UI is built using Streamlit. It allows users to interact with the chatbot and receive answers in real-time.

**Code Snippet:**
```python
import streamlit as st

st.title("Website Content Chatbot")
query = st.text_input("Enter your question:")
if query:
    response = retrieval_chain.run(query)
    st.write(response)
```

---

## Getting Started
### Prerequisites
- Python 3.8+
- Install required packages:
  ```bash
  pip install langchain streamlit
  ```

### Running the Chatbot
1. Clone the repository.
2. Install dependencies using `pip`.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the provided URL in your browser to interact with the chatbot.

---

## Contributing
Feel free to submit issues or pull requests. Contributions are welcome!

---
