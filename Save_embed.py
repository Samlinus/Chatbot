import os
import re
import pickle
from langchain_community.embeddings import HuggingFaceInstructEmbeddings as embed
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


def scrape_and_store_documents(urls):
    """
    Scrape websites and clean their content to store as a list of documents.
    """
    documents = []
    loader = WebBaseLoader(urls)
    try:
        for doc in loader.load():
            if doc.page_content:  # Ensure document has content
                cleaned_content = re.sub(r"\s+", " ", doc.page_content).strip()
                if cleaned_content:
                    documents.append(cleaned_content)
    except Exception as e:
        print(f"Error while scraping websites: {e}")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split large documents into smaller chunks based on specified chunk size and overlap.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for doc in documents:
        try:
            chunks.extend(text_splitter.split_text(doc))
        except Exception as e:
            print(f"Error splitting document: {e}")
    return chunks


def store_embeddings(chunks, vector_path):
    """
    Generate and store embeddings into a FAISS vector database.
    """
    if not chunks:
        print("No chunks to store. Please check the document splitting step.")
        return
    try:
        embedding_model = embed()
        vector_store = FAISS.from_texts(chunks, embedding_model)
        vector_store.save_local(vector_path)
        print(f"Vector store successfully saved to {vector_path}")
    except Exception as e:
        print(f"Error storing embeddings: {e}")


def load_embeddings(vector_path):
    """
    Load embeddings from a local FAISS vector database.
    """
    try:
        vector_store = FAISS.load_local(vector_path, embed(), allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
        return vector_store
    except FileNotFoundError:
        print(f"Vector file not found at {vector_path}. Please ensure the path is correct.")
    except Exception as e:
        print(f"Error loading vector store: {e}")


def main():
    urls = [
        "https://www.gamesville.co.in/",
        "https://www.gamesville.co.in/about-us.html",
        "https://www.gamesville.co.in/squash.html",
        "https://www.gamesville.co.in/Badminton.html",
        "https://www.gamesville.co.in/tennis.html",
        "https://www.gamesville.co.in/Swimming.html",
        "https://www.gamesville.co.in/Gymnasium.html",
        "https://www.gamesville.co.in/tabletennis.html",
        "https://www.gamesville.co.in/Cricket.html",
        "https://www.gamesville.co.in/Silambam.html",
        "https://www.gamesville.co.in/Zumba.html",
        "https://www.gamesville.co.in/Dance.html",
        "https://www.gamesville.co.in/Yoga.html",
        "https://www.gamesville.co.in/Archery.html",
        "https://www.gamesville.co.in/Snooker.html",
        "https://www.gamesville.co.in/football.html",
        "https://www.gamesville.co.in/contact.html"
    ]

    vector_path = r"C:\Users\SamClitusFernando\OneDrive - AVA Software Pvt Ltd\Desktop\FAISS_vector"

    print("Scraping websites...✅")
    documents = scrape_and_store_documents(urls)
    if not documents:
        print("No documents scraped. Exiting.")
        return

    print("Splitting documents into chunks...✅")
    chunks = split_documents(documents)
    if not chunks:
        print("No chunks generated. Exiting.")
        return

    print("Storing embeddings in FAISS vector database...✅")
    store_embeddings(chunks, vector_path)

    print("Loading embeddings from vector database...✅")
    vector_store = load_embeddings(vector_path)
    if vector_store:
        print("Vector store loaded and ready to use!")


if __name__ == "__main__":
    main()
