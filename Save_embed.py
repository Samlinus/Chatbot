import os
import pickle
from langchain_community.embeddings import HuggingFaceInstructEmbeddings as embed
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re

def scrape_and_store_documents(urls):
    documents = []
    loader = WebBaseLoader(urls)
    for doc in loader.load():
        doc = re.sub(r"\s+", " ",doc.page_content).strip()
        documents.append(doc)

    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = []
    for i, doc in enumerate(documents):
        chunks.extend(text_splitter.split_text(doc))
    return chunks

def store_embeddings(chunks, vector_path): #Added vector_path parameter
    embedding_model = embed()
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local(vector_path)
    print(f"Vector store saved to {vector_path}")
        

def load_embeddings(vector_path):
    try:
        return FAISS.load_local(vector_path, embed(), allow_dangerous_deserialization=True)
    except Exception as e:
        print("Error in loading vector..\n", e)

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

    # load_embed = load_embeddings(vector_path)
    # print(load_embed)
    # print("load successfull..")

    print("Scraping websites...✅")
    documents = scrape_and_store_documents(urls)
    # print("Splitting documents into chunks...✅")
    # chunks = split_documents(documents)
    # print("Storing embeddings in FAISS vector database...✅")
    
    # store_embeddings(chunks, vector_path)
    # print("Vector saved...✅")


if __name__ == "__main__":
    main()