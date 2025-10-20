import os
import tempfile
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from docqa.document_loader import DocumentLoader 
from docqa.retriever_function import configure_retriever
import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser


class HistoryAwareRetrievalQAChain:
    """Minimal history-aware retrieval augmented generation chain."""

    def __init__(self, llm, retriever, contextualize_prompt, qa_prompt):
        self.llm = llm
        self.retriever = retriever
        # Parser converts the LLM output into plain text for follow-up usage.
        self._contextualize_chain = contextualize_prompt | llm | StrOutputParser()
        self._qa_prompt = qa_prompt

    def _contextualize_question(self, inputs, config=None):
        chat_history = inputs.get("chat_history") or []
        if not chat_history:
            return inputs["input"]

        # Avoid streaming callbacks during the question rewriting step.
        condense_config = None
        if config:
            condense_config = dict(config)
            condense_config.pop("callbacks", None)

        return self._contextualize_chain.invoke(inputs, config=condense_config)

    def _combine_documents(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, inputs, config=None):
        stand_alone_question = self._contextualize_question(inputs, config=config)
        docs = self.retriever.invoke(stand_alone_question)
        prompt_messages = self._qa_prompt.format_messages(
            context=self._combine_documents(docs),
            input=inputs["input"],
        )

        final_answer = ""
        for chunk in self.llm.stream(prompt_messages, config=config):
            final_answer += chunk.content or ""

        return {"answer": final_answer, "context": docs}

    def stream(self, inputs, config=None):
        stand_alone_question = self._contextualize_question(inputs, config=config)
        docs = self.retriever.invoke(stand_alone_question)
        prompt_messages = self._qa_prompt.format_messages(
            context=self._combine_documents(docs),
            input=inputs["input"],
        )

        yield from self.llm.stream(prompt_messages, config=config)

    def batch(self, inputs_list, config=None):
        return [self.invoke(inputs, config=config) for inputs in inputs_list]

import logging
logging.basicConfig(level=logging.INFO)

@st.cache_resource(show_spinner="Configuring chain...")
def configure_qa_chain(uploaded_files):
    """Read documents, configure retriever, and create the QA chain."""
    # Get Google API key from secrets
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logging.error("GOOGLE_API_KEY environment variable not set.")
        st.error("GOOGLE_API_KEY environment variable not set.")
        st.stop()

    # Read documents
    docs = [] 
    temp_dir = tempfile.mkdtemp() # Create a temporary directory
    try:
        doc_loader = DocumentLoader()
        for file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, file.name)
            with open(temp_file_path, "wb") as f:
                f.write(file.getvalue())
            docs.extend(doc_loader.load_document(temp_file_path))
    finally:
        # Clean up the temporary directory and its contents
        shutil.rmtree(temp_dir)

    # Configure retriever
    retriever = configure_retriever(docs=docs)

    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0,
        streaming=True,
        google_api_key=google_api_key
    )

    # 1. Create a history-aware retriever
    # This prompt helps the LLM rephrase the user's question to be a standalone question
    # based on the chat history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    # 2. Create a chain to answer questions based on retrieved documents
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "{input}")])
    return HistoryAwareRetrievalQAChain(
        llm=llm,
        retriever=retriever,
        contextualize_prompt=contextualize_q_prompt,
        qa_prompt=qa_prompt,
    )

if __name__ == '__main__':
    st.set_page_config(page_title="LangChain: Chat with Documents (Gemini)", page_icon=":books:")
    st.title("LangChain: Chat with Documents (Gemini) 📚")
    

    # Instantiate the document loader once
    doc_loader = DocumentLoader()
    uploaded_files = st.sidebar.file_uploader(
        label="Upload your documents here", 
        type=list(doc_loader.supported_extensions.keys()),
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload at least one document to proceed.")
        st.stop()

    # Initialize or retrieve chat history from session state.
    # The old chain managed memory internally, but with LCEL, it's better
    # to manage it explicitly in the application state.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! How can I help you with your documents?")]

    # Display past messages
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)

    # Configure and retrieve the QA chain from cache
    qa_chain = configure_qa_chain(uploaded_files)

    user_query = st.chat_input("Ask a question about your documents")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("assistant"):
            stream_handler = StreamlitCallbackHandler(st.container())
            # Pass the user query and chat history to the LCEL chain
            response = qa_chain.invoke(
                {"input": user_query, "chat_history": st.session_state.chat_history}, 
                {"callbacks": [stream_handler]})
            
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            st.rerun() # Rerun to display the new message immediately
