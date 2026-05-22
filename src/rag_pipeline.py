import logging
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate  # replace ChatPromptTemplate

from embedder import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpecLensPipeline:
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the RAG pipeline: loads the vector store and the local LLM.
        """
        # When device_map="auto" is used, HuggingFace manages device placement itself.
        # We only pass device= to pipeline() in CPU mode (-1); GPU mode leaves it unset.
        self.use_gpu = use_gpu

        # Load the FAISS vector store
        logging.info("Loading local FAISS vector store...")
        self.vector_manager = VectorStoreManager(use_gpu=use_gpu)
        self.vector_store = self.vector_manager.load_index()

        # MMR (Maximal Marginal Relevance) retriever: fetches k=8 candidates then re-ranks
        # them by diversity, returning the 5 most varied and relevant chunks.
        # This prevents the retriever from returning 4 near-identical procedural paragraphs
        # (which is what happened with plain similarity search on spec text).
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,       # Final chunks passed to the LLM
                "fetch_k": 8, # Candidate pool MMR selects from
            }
        )

        # Initialize the local LLM (SmolLM2 from HuggingFace)
        self.model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        logging.info(f"Initializing local LLM: {self.model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto" if use_gpu else None
        )

        # device= is only passed in CPU mode (-1); when device_map="auto" is active,
        # HuggingFace has already placed the model and passing device= raises a conflict error.
        pipe_kwargs = dict(
            max_new_tokens=512, # increase this if the model's answer seems truncated
            temperature=0.1,  # Low temperature = less hallucination
            do_sample=True,
            return_full_text=False,  # ← only return generated tokens, not the prompt
        )
        if not use_gpu:
            pipe_kwargs["device"] = -1

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **pipe_kwargs
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.prompt = self._build_prompt()

        logging.info("Assembling LangChain RAG pipeline...")
        self.qa_chain = self._build_chain()

    def _build_prompt(self):
        # SmolLM2 uses a specific chat template that must be applied via the tokenizer,
        # not via LangChain's ChatPromptTemplate. Using ChatPromptTemplate causes the
        # model to echo the system prompt rather than generate a response.
        template = (
            "<|im_start|>system\n"
            "You are a 3GPP specification assistant. "
            "The retrieved context may contain multiple message definitions and ASN.1 code. "
            "Find the section that directly answers the question and summarise ONLY that part. "
            "If the answer is not present, say: 'I cannot answer this based on the retrieved 3GPP specifications.' "
            "Do not describe unrelated messages. Do not reproduce ASN.1 code.\n\n"
            "Retrieved Context:\n{context}"
            "<|im_end|>\n"
            "<|im_start|>user\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return PromptTemplate(input_variables=["context", "input"], template=template)

    @staticmethod
    def format_docs(docs):
        chunks = []
        for doc in docs:
            # Strip ASN.1 blocks — they consume context window without adding semantic value
            text = re.sub(r"-- ASN1START.*?-- ASN1STOP", "", doc.page_content, flags=re.DOTALL)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                chunks.append(text)
        return "\n\n".join(chunks)

    @staticmethod
    def rephrase_for_retrieval(question: str) -> str:
        """
        Rewrites interrogative questions into declarative form to better match
        the assertive language used in 3GPP spec prose.

        3GPP specs never say "What is the purpose of X?" — they say
        "The X message is used to..." or "X is defined as...".
        Plain similarity search on interrogative queries tends to surface
        procedural mentions instead of definitional sections.

        Examples:
            "What is the purpose of RRCReconfiguration?"
            -> "RRCReconfiguration purpose definition used for"

            "What triggers a Radio Link Failure?"
            -> "Radio Link Failure triggers conditions causes"

            "How does the UE handle T310?"
            -> "T310 UE handling procedure behavior"
        """
        q = question.strip().rstrip("?")
        q_lower = q.lower()

        # "What is the purpose of X" / "What does X do"
        m = re.match(r"what is the purpose of (.+)", q_lower)
        if m:
            term = q[len("what is the purpose of "):]
            return f"{term} purpose definition used for description"

        m = re.match(r"what does (.+?) do", q_lower)
        if m:
            term = m.group(1)
            return f"{term} function purpose behavior"

        # "What is X" / "What are X"
        m = re.match(r"what (?:is|are) (.+)", q_lower)
        if m:
            term = q[re.match(r"what (?:is|are) ", q_lower).end():]
            return f"{term} definition description"

        # "What triggers / causes X"
        m = re.match(r"what (?:triggers?|causes?) (.+)", q_lower)
        if m:
            term = m.group(1)
            return f"{term} triggers causes conditions initiation"

        # "How does X work" / "How does the UE handle X"
        m = re.match(r"how does (?:the )?(.+?)(?:\s+work| handle (.+))?$", q_lower)
        if m:
            subject = m.group(1)
            obj = m.group(2) or ""
            return f"{subject} {obj} procedure behavior handling".strip()

        # "When is X used" / "When does X occur"
        m = re.match(r"when (?:is|does) (.+?)(?:\s+used|\s+occur)?$", q_lower)
        if m:
            term = m.group(1)
            return f"{term} conditions when applicable"

        # Default: return unchanged
        return question

    def _build_chain(self):
        """
        Combines the LLM, prompt, and retriever using LCEL.

        Query flow:
          user_question
            -> rephrase_for_retrieval()   (improves FAISS hit quality)
            -> MMR retriever              (k=5 diverse chunks)
            -> format_docs + original question -> prompt -> LLM
        """
        # Rewrite the query before it hits FAISS, but keep the original
        # question for the LLM prompt so the answer addresses what was asked.
        def retrieve_with_rephrased_query(user_question: str):
            rephrased = SpecLensPipeline.rephrase_for_retrieval(user_question)
            logging.info(f"Retrieval query (rephrased): '{rephrased}'")
            docs = self.retriever.invoke(rephrased)
            return {"context": docs, "input": user_question}

        generation_step = (
            {
                "context": lambda x: SpecLensPipeline.format_docs(x["context"]),
                "input": lambda x: x["input"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        lc_chain = (
            RunnableLambda(retrieve_with_rephrased_query)
            .assign(answer=generation_step)
        )

        return lc_chain

    def query(self, user_question: str) -> dict:
        """
        Executes a query against the RAG pipeline.

        Returns a dict with keys:
            "input"   - original user question
            "context" - List[Document] retrieved chunks
            "answer"  - LLM-generated answer string
        """
        logging.info(f"Executing query: '{user_question}'")
        return self.qa_chain.invoke(user_question)


if __name__ == "__main__":
    try:
        rag = SpecLensPipeline(use_gpu=False)

        print("\n" + "=" * 50)
        print(" SpecLens Engine Ready ")
        print("=" * 50 + "\n")

        test_query = "What is the purpose of the RRCReconfiguration message?"
        print(f"User Query: {test_query}\n")

        result = rag.query(test_query)

        print("\n--- LLM ANSWER ---")
        print(result["answer"])

        print("\n--- RETRIEVED SOURCES ---")
        for i, doc in enumerate(result["context"]):
            page = doc.metadata.get("page", "Unknown Page")
            print(f"[{i + 1}] 3GPP TS 38.331 - Page {page}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")