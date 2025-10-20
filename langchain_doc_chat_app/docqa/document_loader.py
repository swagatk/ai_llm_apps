import os
from typing import Any 
import logging
import pathlib 
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader
)

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str| list[str], **kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")

class DocumentLoaderException(Exception):
    """Custom exception for document loader errors."""
    pass


class DocumentLoader:
    def __init__(self):
        self.supported_extensions = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".epub": EpubReader,
            ".doc": UnstructuredWordDocumentLoader,
            ".odt": UnstructuredWordDocumentLoader,
            ".docx": UnstructuredWordDocumentLoader,
        }

    def load_document(self, file_path: str) -> list[Document]:
        ext = pathlib.Path(file_path).suffix.lower()
        loader = self.supported_extensions.get(ext) 

        if not loader:
            raise DocumentLoaderException(
                f"Invalid Extension type {ext}, cannot load this file"
            )
        loader = loader(file_path)
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents from {file_path}") 
        return docs


if __name__ == '__main__':
    # Example usage:
    # Create dummy files for testing
    with open("example.txt", "w") as f:
        f.write("This is a sample text document.")
    
    # You would need a PDF and DOCX file for full testing
    # For instance, create a dummy PDF:
    # from reportlab.pdfgen import canvas
    # c = canvas.Canvas("example.pdf")
    # c.drawString(100, 750, "This is a sample PDF document.")
    # c.save()

    # For DOCX, you'd typically create it with a word processor or a library like python-docx

    # Load a text document
    loader_instance = DocumentLoader()
    txt_docs = loader_instance.load_document("example.txt")
    print(f"Loaded text document: {txt_docs[0].page_content}")

    # Load a PDF document (uncomment if you have example.pdf)
    # pdf_docs = loader_instance.load_document("example.pdf")
    # print(f"Loaded PDF document: {pdf_docs[0].page_content}")

    # Load a DOCX document (uncomment if you have example.docx)
    # docx_docs = loader_instance.load_document("example.docx")
    # print(f"Loaded DOCX document: {docx_docs[0].page_content}")

    os.remove("example.txt")