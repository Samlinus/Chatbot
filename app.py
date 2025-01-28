import streamlit as st
from chatbot import ChatbotHandler  # Assuming your ChatbotHandler is in chatbot.py

# Initialize the ChatbotHandler outside the Streamlit app
vector_store_path = r"C:\Users\SamClitusFernando\OneDrive - AVA Software Pvt Ltd\Desktop\FAISS_vector"  # Replace with your actual path
chatbot = ChatbotHandler(vector_store_path)


st.title("Gamesville Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hii, How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chatbot.get_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})
                st.write(f"An error occurred: {e}") #Display error in chat