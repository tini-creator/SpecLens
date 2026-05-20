import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Import the vector store manager
from embedder import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpecLensPipeline:
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the RAG pipeline: loads the vector store and the local LLM.
        """
        self.device = 0 if use_gpu else -1  # 0 for CUDA (RTX 3090), -1 for CPU

        # Load the FAISS Retriever
        logging.info("Loading local FAISS vector store...")
        self.vector_manager = VectorStoreManager()
        self.vector_store = self.vector_manager.load_index()

        # Configure retriever to fetch the top 4 most relevant 3GPP chunks
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        # 2. Initialize the Local LLM (SmolLM2 from HuggingFace)
        # We use SmolLM2-1.7B-Instruct as it is highly efficient and capable of following instructions
        self.model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        logging.info(f"Initializing local LLM: {self.model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto" if use_gpu else None
        )

        # Create a HuggingFace text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Keep answers concise
            temperature=0.1,  # Low temperature = less hallucination
            do_sample=True,
            device=self.device
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Architect the 3GPP System Prompt
        self.prompt = self._build_prompt()

        # Construct the LangChain Retrieval Chain
        logging.info("Assembling LangChain RAG pipeline...")
        self.qa_chain = self._build_chain()

    def _build_prompt(self) -> ChatPromptTemplate:
        """
        Engineers the strict system prompt to prevent telecom hallucinations.
        """
        system_prompt = (
            "You are a Senior 5G Firmware Diagnostics Engineer. "
            "Use ONLY the following retrieved 3GPP specification context to answer the user's question. "
            "If the answer is not contained in the context, you must explicitly state: 'I cannot answer this based on the retrieved 3GPP specifications.' "
            "Do not hallucinate technical parameters, timers, or Information Elements. "
            "\n\n"
            "Retrieved Context:\n{context}"
        )

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

    @staticmethod
    # Helper to convert the retrieved document chunks into a single string for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """
        Combines the LLM, Prompt, and Retriever using pure LCEL.
        This bypasses the legacy 'chains' module entirely for a more robust architecture.
        """

        # Retrieve the documents and pass the user's question straight through
        setup_and_retrieval = RunnableParallel({
            "context": self.retriever,
            "input": RunnablePassthrough()
        })

        # Format the inputs, pass them into the prompt, and generate the LLM response
        generation_step = (
                {
                    "context": lambda x: self.format_docs(x["context"]),
                    "input": lambda x: x["input"]
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

        # Combine both steps so the final output dictionary contains BOTH the answer and the source documents
        lc_chain = setup_and_retrieval.assign(answer=generation_step)

        return lc_chain

    def query(self, user_question: str) -> dict:
        """
        Executes a query against the RAG pipeline.
        """
        logging.info(f"Executing query: '{user_question}'")
        # With LCEL, we invoke the chain directly with the raw string
        response = self.qa_chain.invoke(user_question)
        return response


if __name__ == "__main__":
    # --- Local Testing Block ---
    try:
        # Set use_gpu=True to leverage GPU, or False if running on a standard laptop
        rag = SpecLensPipeline(use_gpu=False)

        print("\n" + "=" * 50)
        print(" SpecLens Engine Ready ")
        print("=" * 50 + "\n")

        test_query = "What triggers a Radio Link Failure (RLF)?"
        print(f"User Query: {test_query}\n")

        result = rag.query(test_query)

        print("\n--- LLM ANSWER ---")
        print(result["answer"])

        print("\n--- RETRIEVED SOURCES ---")
        for i, doc in enumerate(result["context"]):
            # Extract the page number from the metadata we saved in document_parser.py
            page = doc.metadata.get('page', 'Unknown Page')
            print(f"[{i + 1}] 3GPP TS 38.331 - Page {page}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")