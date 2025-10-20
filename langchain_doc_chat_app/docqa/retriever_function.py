
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_compressors import EmbeddingFilter
from langchain_core.documents import Document
from langchain_core.retrievers import ContextualCompressionRetriever 

def configure_retriever(docs: list[Document], use_compression: bool = True) -> BaseRetriever:
    """
    Configures and returns a retriever for the given documents.

    Args:
        docs: A list of documents to be indexed.
        use_compression: Whether to use a contextual compression retriever.

    Returns:
        A configured BaseRetriever instance.
    """
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a vector store from the split documents
    vector_store = DocArrayInMemorySearch.from_documents(split_docs, embeddings)

    # Create and return a retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    if not use_compression:
        return retriever

    # Set up document compressor
    embedding_filter = EmbeddingFilter.from_embeddings(embeddings, similarity_threshold=0.8)
    return ContextualCompressionRetriever(
        base_compressor=embedding_filter, 
        base_retriever=retriever
    )
    