import os
import argparse
import logging
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import the parser
from document_parser import TelecomDocumentParser

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorStoreManager:
    def __init__(self, index_dir: str = "vector_store/", use_gpu: bool = False):
        """
        Initializes the embedding model and index directory.

        Args:
            index_dir: Path where the FAISS index is saved/loaded.
            use_gpu: If True, runs the embedding model on CUDA. Defaults to False
                     (CPU is sufficient for inference; GPU matters mainly during bulk indexing).
        """
        self.index_dir = index_dir

        # We use a lightweight, highly efficient local embedding model.
        # 'all-MiniLM-L6-v2' maps sentences & paragraphs to a 384 dimensional dense vector space.
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if use_gpu else "cpu"
        logging.info(f"Initializing embedding model: {self.model_name} on {device} (This runs 100% locally)")

        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}  # Normalization improves cosine similarity search

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def build_index(self, documents: List[Document]):
        """
        Ingests document chunks, generates embeddings, and saves the FAISS index to disk.
        """
        if not documents:
            raise ValueError("No documents provided to build the index.")

        logging.info(f"Building FAISS index for {len(documents)} document chunks...")
        logging.info("This may take a few minutes depending on your hardware.")

        # Create the FAISS vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)

        # Ensure the output directory exists
        os.makedirs(self.index_dir, exist_ok=True)

        # Save the index locally
        vector_store.save_local(self.index_dir)
        logging.info(f"Successfully saved FAISS index to {self.index_dir}")

        return vector_store

    def load_index(self) -> FAISS:
        """
        Loads an existing FAISS index from disk.
        """
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"No FAISS index found at {self.index_dir}. Please build it first.")

        logging.info(f"Loading FAISS index from {self.index_dir}...")

        # We must set allow_dangerous_deserialization=True because we are loading a local pickle file.
        # This is safe here because we generated the file ourselves locally.
        vector_store = FAISS.load_local(
            self.index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store


if __name__ == "__main__":
    # CLI configuration
    parser = argparse.ArgumentParser(description="Build or test the FAISS Vector Store for SpecLens.")
    parser.add_argument("--build-index", action="store_true", help="Parse the PDF and build the FAISS index.")
    parser.add_argument("--pdf-path", type=str, default="data/raw/ts_38331.pdf", help="Path to the 3GPP PDF.")

    args = parser.parse_args()

    vector_manager = VectorStoreManager()

    if args.build_index:
        # Parse the documents
        doc_parser = TelecomDocumentParser()
        try:
            chunks = doc_parser.load_and_split_pdf(args.pdf_path)

            # Build and save the index
            vector_manager.build_index(chunks)
        except Exception as e:
            logging.error(f"Failed to build index: {e}")
            print(f"\nEnsure you have placed your PDF at: {args.pdf_path}")

    else:
        # If run without arguments, perform a quick sanity check to see if the index loads
        try:
            store = vector_manager.load_index()
            print("\nFAISS index loaded successfully! It is ready to be queried by the RAG pipeline.")
            print("Run with --build-index to generate a new index from your PDF.")
        except FileNotFoundError:
            print("\nIndex not found. Run: python src/embedder.py --build-index")