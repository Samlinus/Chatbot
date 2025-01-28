import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings as embed

load_dotenv()

class ChatbotHandler:
    def __init__(self, vector_store_path: str):
        self.embeddings = embed()  # Initialize embeddings
        self.vector_store_path = vector_store_path #Store the path, not the loaded vectorstore
        self.prompt = self._set_template()
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"))

    def _set_template(self) -> ChatPromptTemplate:
        """Sets the prompt template."""
        template = """
            You are an AI chatbot owned by gamesville sports club tuticorin.
            <instruction>
            * Use the given context to answer the question.
            * If the answer is not explicitly mentioned in the context, try to provide a helpful and relevant response based on your general knowledge.
            * Do not hallucinate
            * You are an I chantbot developed only to answer user queries.
            </instruction>

            Context: {context}
            Human input: {input}"""
        return ChatPromptTemplate.from_template(template)

    def get_response(self, query: str) -> str:
        """Gets the chatbot response for a given query."""
        try:
            # Load the vector store *only when needed* for a query
            vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever() #Create retriever within function
            qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
            retrieval_chain = create_retrieval_chain(retriever, qa_chain)
            answer = retrieval_chain.invoke({"input": query})
            return answer["answer"]
        except Exception as e:
            print(f"An error occurred during RAG: {e}")
            return f"An error occurred: {e}"