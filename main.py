
import os
from dotenv import load_dotenv
import hashlib
import pickle
from functools import lru_cache
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient

load_dotenv()
# Load Gemini API Key from file
# with open("API_KEY.txt", "r") as f:
#     GEMINI_API_KEY = f.read().strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

# Hugging Face config
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ Custom LangChain-compatible wrapper
class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model: str, token: str):
        self.client = InferenceClient(model=model, token=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.client.feature_extraction(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text)


class RAGPipeline:
    def __init__(self, pdf_path, use_cache=True):
        self.pdf_path = pdf_path
        self.use_cache = use_cache
        self.cache_dir = "rag_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # ✅ Initialize wrapped embedding model
        self.embedding_model = HFInferenceEmbeddings(model=HF_MODEL, token=HF_TOKEN)

        self._initialize_components()

    def _get_cache_path(self, suffix: str) -> str:
        file_hash = hashlib.md5(self.pdf_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{file_hash}_{suffix}.pkl")

    def _initialize_components(self):
        vectorstore_cache = self._get_cache_path("vectorstore")
        chunks_cache = self._get_cache_path("chunks")

        if self.use_cache and os.path.exists(vectorstore_cache) and os.path.exists(chunks_cache):
            self._load_from_cache()
        else:
            self._create_components()
            if self.use_cache:
                self._save_to_cache()

        self._initialize_llm_chains()

    def _load_from_cache(self):
        with open(self._get_cache_path("chunks"), 'rb') as f:
            self.text_chunks = pickle.load(f)

        vectorstore_path = self._get_cache_path("vectorstore")
        self.vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_type="mmr")

    def _save_to_cache(self):
        with open(self._get_cache_path("chunks"), 'wb') as f:
            pickle.dump(self.text_chunks, f)
        self.vectorstore.save_local(self._get_cache_path("vectorstore"))

    def _create_components(self):
        self.documents = PyMuPDFLoader(self.pdf_path).load()
        self.text_chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(self.documents)

        self.vectorstore = FAISS.from_documents(self.text_chunks, self.embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_type="mmr")

    def _initialize_llm_chains(self):
        self.llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )

#         custom_prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
# You are a professional insurance document analyst. Your task is to answer user queries based strictly on the provided insurance policy document context.

# Guidelines:
# - Use ONLY the information in the context.
# - Be accurate and formal.
# - Include policy details (terms, monetary values, legal references).
# - Start with "Yes" or "No" if applicable.
# - If info not present, say: "Information not available in the provided document."
# - Limit to ONE sentence.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#         )

#         custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a professional insurance document analyst.

# Your task is to answer user questions based strictly on the context from the provided insurance policy document.

# Follow these rules:

# - Only use information explicitly mentioned in the context.
# - Do NOT make assumptions or use external knowledge.
# - If the answer is clearly stated, begin with "Yes" or "No" followed by a precise explanation.
# - If the answer is not clearly stated, reply with: "Information not available in the provided document."
# - Include exact details if present: durations, monetary limits, conditions, or exclusions.
# - Avoid vague or partial answers.
# - Respond in a clear, natural tone.
# - Your answer must be in **ONE** concise sentence.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )
        custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional insurance document analyst.

Answer the question strictly using only the information provided in the context below. Do not assume anything not explicitly stated.

Follow these rules:

- Use *only* the context. No external knowledge or assumptions.
- Be precise with durations, monetary values, age limits, policy terms, and conditions.
- If the answer is clearly stated, respond with "Yes" or "No" followed by the exact supporting clause from the document.
- If the answer is not explicitly present, respond exactly with: "Information not available in the provided document."
- Include full clause-level details such as number of events, continuity terms, facility criteria, or legal references if available.
- Do not rephrase or simplify — use exact language from the context where applicable.
- Limit your response to *one clear, complete, and compliant sentence*.
- Maintain a formal, objective tone.

Context:
{context}

Question:
{question}

Answer:
"""
)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_model,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=False
        )

    @lru_cache(maxsize=500)
    def _cached_ask(self, question_hash: str, question: str) -> str:
        return self.qa_chain.run(question)

    def ask(self, question: str) -> str:
        question_hash = hashlib.md5(question.encode()).hexdigest()
        return self._cached_ask(question_hash, question)

    def batch_ask(self, questions: List[str]) -> List[str]:
        return [self.ask(q) for q in questions]

    def clear_cache(self):
        self._cached_ask.cache_clear()

    def get_cache_info(self):
        return {
            "question_cache_info": self._cached_ask.cache_info()._asdict(),
            "vectorstore_cached": os.path.exists(self._get_cache_path("vectorstore")),
            "chunks_cached": os.path.exists(self._get_cache_path("chunks"))
        }
