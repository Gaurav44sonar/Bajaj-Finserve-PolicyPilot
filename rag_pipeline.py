# import os
# from dotenv import load_dotenv
# import hashlib
# import pickle
# from functools import lru_cache
# from typing import List
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain.embeddings.base import Embeddings
# from huggingface_hub import InferenceClient

# load_dotenv()
# # Load Gemini API Key from file
# # with open("API_KEY.txt", "r") as f:
# #     GEMINI_API_KEY = f.read().strip()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("Missing GEMINI_API_KEY environment variable")

# genai.configure(api_key=GEMINI_API_KEY)

# # Hugging Face config
# HF_TOKEN = os.getenv("HF_TOKEN")
# HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# # ✅ Custom LangChain-compatible wrapper
# class HFInferenceEmbeddings(Embeddings):
#     def __init__(self, model: str, token: str):
#         self.client = InferenceClient(model=model, token=token)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [self.client.feature_extraction(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return self.client.feature_extraction(text)


# class RAGPipeline:
#     def __init__(self, pdf_path, use_cache=True):
#         self.pdf_path = pdf_path
#         self.use_cache = use_cache
#         self.cache_dir = "rag_cache"
#         os.makedirs(self.cache_dir, exist_ok=True)

#         # ✅ Initialize wrapped embedding model
#         self.embedding_model = HFInferenceEmbeddings(model=HF_MODEL, token=HF_TOKEN)

#         self._initialize_components()

#     def _get_cache_path(self, suffix: str) -> str:
#         file_hash = hashlib.md5(self.pdf_path.encode()).hexdigest()
#         return os.path.join(self.cache_dir, f"{file_hash}_{suffix}.pkl")

#     def _initialize_components(self):
#         vectorstore_cache = self._get_cache_path("vectorstore")
#         chunks_cache = self._get_cache_path("chunks")

#         if self.use_cache and os.path.exists(vectorstore_cache) and os.path.exists(chunks_cache):
#             self._load_from_cache()
#         else:
#             self._create_components()
#             if self.use_cache:
#                 self._save_to_cache()

#         self._initialize_llm_chains()

#     def _load_from_cache(self):
#         with open(self._get_cache_path("chunks"), 'rb') as f:
#             self.text_chunks = pickle.load(f)

#         vectorstore_path = self._get_cache_path("vectorstore")
#         self.vectorstore = FAISS.load_local(
#             vectorstore_path,
#             embeddings=self.embedding_model,
#             allow_dangerous_deserialization=True
#         )
#         self.retriever = self.vectorstore.as_retriever(search_type="mmr")

#     def _save_to_cache(self):
#         with open(self._get_cache_path("chunks"), 'wb') as f:
#             pickle.dump(self.text_chunks, f)
#         self.vectorstore.save_local(self._get_cache_path("vectorstore"))

#     def _create_components(self):
#         self.documents = PyMuPDFLoader(self.pdf_path).load()
#         self.text_chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(self.documents)

#         self.vectorstore = FAISS.from_documents(self.text_chunks, self.embedding_model)
#         self.retriever = self.vectorstore.as_retriever(search_type="mmr")

#     def _initialize_llm_chains(self):
#         self.llm_model = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash",
#             google_api_key=GEMINI_API_KEY,
#             temperature=0.0
#         )

# #         custom_prompt = PromptTemplate(
# #             input_variables=["context", "question"],
# #             template="""
# # You are a professional insurance document analyst. Your task is to answer user queries based strictly on the provided insurance policy document context.

# # Guidelines:
# # - Use ONLY the information in the context.
# # - Be accurate and formal.
# # - Include policy details (terms, monetary values, legal references).
# # - Start with "Yes" or "No" if applicable.
# # - If info not present, say: "Information not available in the provided document."
# # - Limit to ONE sentence.

# # Context:
# # {context}

# # Question:
# # {question}

# # Answer:
# # """
# #         )

#     ## New prompt template with more advancements
        
# #         custom_prompt = PromptTemplate(
# #     input_variables=["context", "question"],
# #     template="""
# # You are a highly specialized AI assistant trained to analyze and extract information from insurance, legal, and compliance documents.

# # Your task is to answer user questions based **strictly and only** on the retrieved context provided below.
# # You are an expert insurance policy assistant. Your task is to answer the user's question using only the content provided in the policy document context.



# # ---
# # RETRIEVED CONTEXT:
# # {context}
# # ---

# # QUESTION:
# # {question}

# # GUIDELINES:
# # 1. Analyze the entire context and identify **only the most relevant parts** related to the question.
# # 2. Provide a precise, focused, and fact-based answer using **exact language, figures, terms, and conditions** found in the document.
# # 3. Convert number words (e.g., "thirty", "fifty-six") into digits (e.g., "30", "56") in your answer.
# # 4. Do **not summarize the entire context** — answer based only on **parts directly relevant** to the question.
# # 5. If multiple sections apply, mention their section numbers clearly (e.g., "As per Context 2.4 and 5.1...").
# # 6. If the question cannot be answered using the provided context, respond exactly with: **"Information not available in the provided document."**
# # 7. Use exact policy terms and values when available.

# # 8. Avoid assumptions, opinions, or any unsupported statements.
# # 9. Respond in one complete, formal sentence.
# # 10. Keep the response short and to the point — ideally under 50 words unless essential information must be conveyed.

# # ANSWER:
# # """
# # )

# #         custom_prompt = PromptTemplate(
# #     input_variables=["context", "question"],
# #     template="""
# # You are an expert insurance policy assistant. Your task is to answer the user's question using only the content provided in the policy document context.

# # Instructions:
# # - Respond in one complete, formal sentence.
# # - Use clear, human-readable language appropriate for both legal and customer communication.
# # - Do not explain or elaborate beyond the document.
# # - Use exact policy terms and values when available.
# # - If the answer is not found in the context, reply: "The document does not provide information about this."

# # Context:
# # {context}

# # Question:
# # {question}

# # Answer:
# # """
# # )

#         # 1. Custom prompt Shooter
#         custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an expert document analyst specializing in insurance, legal, and compliance policy. You will answer questions using ONLY the provided context from the document retrieval system.

# RETRIEVED CONTEXT:
# {context}

# QUESTION: {question}

# INSTRUCTIONS:
# 1. Answer the question using ONLY information from the retrieved context above
# 2. Be precise and factual - include specific numbers, dates, percentages, and conditions when mentioned
# 3. If the context contains relevant information, provide a comprehensive answer
# 4. If the information is insufficient or not available in the context, clearly state "Information not available in the provided document"
# 5. Structure your answer clearly and logically
# 6. Reference specific context sections when making claims (e.g., "According to Context 1...")
# 7. Keep your response focused and under 600 words

# ANSWER:
# """
# )

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm_model,
#             chain_type="stuff",
#             retriever=self.retriever,
#             chain_type_kwargs={"prompt": custom_prompt},
#             return_source_documents=False
#         )

#     @lru_cache(maxsize=500)
#     def _cached_ask(self, question_hash: str, question: str) -> str:
#         return self.qa_chain.run(question)

#     def ask(self, question: str) -> str:
#         question_hash = hashlib.md5(question.encode()).hexdigest()
#         return self._cached_ask(question_hash, question)

#     def batch_ask(self, questions: List[str]) -> List[str]:
#         return [self.ask(q) for q in questions]

#     def clear_cache(self):
#         self._cached_ask.cache_clear()

#     def get_cache_info(self):
#         return {
#             "question_cache_info": self._cached_ask.cache_info()._asdict(),
#             "vectorstore_cached": os.path.exists(self._get_cache_path("vectorstore")),
#             "chunks_cached": os.path.exists(self._get_cache_path("chunks"))
#         }


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
        self.retriever = self.vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":12})

    def _save_to_cache(self):
        with open(self._get_cache_path("chunks"), 'wb') as f:
            pickle.dump(self.text_chunks, f)
        self.vectorstore.save_local(self._get_cache_path("vectorstore"))

    def _create_components(self):
        self.documents = PyMuPDFLoader(self.pdf_path).load()
        self.text_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(self.documents)

        self.vectorstore = FAISS.from_documents(self.text_chunks, self.embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":12})

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

#         # # 1. Custom prompt Shooter
#         custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an expert document analyst specializing in insurance, legal, and compliance documents. You will answer questions using ONLY the provided context from the document retrieval system.

# RETRIEVED CONTEXT:
# {context}

# QUESTION: {question}

# INSTRUCTIONS:
# 1. Answer the question using ONLY information from the retrieved context above
# 2. Be precise and factual - include specific numbers, dates, percentages, and conditions when mentioned
# 3. If the context contains relevant information, provide a comprehensive answer
# 4. If the information is insufficient or not available in the context, clearly state "Information not available in the provided document"
# 5. Structure your answer clearly and logically
# 6. Reference specific context sections when making claims (e.g., "According to Context 1...")
# 7. Keep your response focused and under 500 words

# ANSWER:
# """
# )

## Sarvesh
#         custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a professional insurance document analyst.

# Answer the question strictly using only the information provided in the context below. Do not assume anything not explicitly stated.

# Follow these rules:

# - Use *only* the context. No external knowledge or assumptions.
# - Be precise with durations, monetary values, age limits, policy terms, and conditions.
# - If the answer is clearly stated, respond with "Yes" or "No" followed by the exact supporting clause from the document.
# - If the answer is not explicitly present, respond exactly with: "Information not available in the provided document."
# - Include full clause-level details such as number of events, continuity terms, facility criteria, or legal references if available.
# - Do not rephrase or simplify — use exact language from the context where applicable.
# - Limit your response to *one clear, complete, and compliant sentence*.
# - Maintain a formal, objective tone.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

## Me
       

        custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a highly specialized AI assistant trained to analyze and extract information from insurance, legal, and compliance policies.

Your task is to answer user questions based **strictly and only** on the retrieved context provided below.

---
RETRIEVED CONTEXT:
{context}
---

QUESTION:
{question}

GUIDELINES:
1. Analyze the entire context and identify **only the most relevant parts** related to the question.
2. Provide a precise, focused, and fact-based answer using **exact language, figures, terms, and conditions** found in the document.
3. Convert number words (e.g., "thirty", "fifty-six") into digits (e.g., "30", "56") in your answer.
4. Do **not summarize the entire context** — answer based only on **parts directly relevant** to the question.
5. If the question cannot be answered using the provided context, respond exactly with: **"Information not available in the provided document."**
6. Avoid assumptions, opinions, or any unsupported statements.
7. Keep the response short and to the point.

ANSWER:
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

