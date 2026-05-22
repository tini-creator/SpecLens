import os
import sys
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# sys.path extension lets sibling modules be imported when running this file directly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class TelecomDocumentParser:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        """
        Initializes the parser with settings optimized for dense technical standards.

        Args:
            chunk_size: Target size of each text chunk. 1500 is chosen to comfortably
                        fit an entire 3GPP Information Element (IE) or section description.
            chunk_overlap: Overlap prevents context loss if a sentence or list is split
                           across two chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # The core of the idea is in the separators. We instruct the splitter to try splitting
        # by double newlines (paragraphs/sections) first, then single newlines,
        # before ever breaking mid-sentence.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n– ",  # 3GPP message/IE header (en-dash) — split HERE, keep header+body together
                "\n\n",
                "\n",
                " ",
                "",
            ]
        )

    def load_and_split_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF and splits it into semantic chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

        logging.info(f"Loading 3GPP specification from: {file_path}")

        try:
            # PyPDFLoader extracts text while preserving the page number in the metadata,
            # which is crucial for citing sources in the final LLM response.
            # pip install pypdf is essential
            loader = PyPDFLoader(file_path)
            raw_documents = loader.load()
            logging.info(f"Successfully loaded {len(raw_documents)} pages.")

            # Apply the custom telecom chunking strategy
            logging.info("Applying semantic chunking to preserve Information Elements (IEs)...")
            chunked_documents = self.text_splitter.split_documents(raw_documents)

            logging.info(f"Split document into {len(chunked_documents)} manageable chunks.")
            return chunked_documents

        except Exception as e:
            logging.error(f"Failed to parse document: {str(e)}")
            raise


if __name__ == "__main__":
    # --- Local Testing Block ---
    # Run `python src/document_parser.py` directly to verify
    # the chunking strategy before connecting it to the vector database.

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data/raw", "ts_38331.pdf"))
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)
    if not os.path.exists(RAW_FILE):
        print(f"Please place your 3GPP PDF at {RAW_FILE} to run the test.")
    else:
        parser = TelecomDocumentParser()
        chunks = parser.load_and_split_pdf(RAW_FILE)

        # Print out the first 3 chunks to manually inspect the quality
        print("\n--- INSPECTING FIRST 3 CHUNKS ---\n")
        for i, chunk in enumerate(chunks[:3]):
            print(f"CHUNK {i + 1} (Page: {chunk.metadata.get('page', 'Unknown')}):")
            print("-" * 40)
            print(chunk.page_content)
            print("-" * 40 + "\n")