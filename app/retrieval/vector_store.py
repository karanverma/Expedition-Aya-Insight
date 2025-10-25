"""
Vector database operations for document storage and retrieval.
"""
from typing import List, Dict, Any, Optional
import os

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    COHERERANK_TOPN,
    VECTOSTORE_TOPK,
)
import cohere


class Retriever:
    """
    Wrapper for vector database operations including document storage,
    similarity search, and reranking of results.
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        """
        Initialize the retriever with embedding model and text splitter.

        Args:
            model: The embedding model name to use for vectorization
        """
        # Get the API key from environment variable
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        self.cohere_client = cohere.Client(api_key=api_key)
        self.faiss = None
        self.embedding_model = CohereEmbeddings(model=model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def create_from_documents(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create vector store from extracted document texts.

        Args:
            extraction_results: List of dictionaries containing filename and extracted text

        Returns:
            Updated extraction results with chunk size information
        """
        chunks = []
        filename = result['filename']
        text = result['text']
        if text:
            document = Document(
                page_content=text,
                metadata={"filename": filename}
            )
            doc_chunks = self.text_splitter.split_documents([document])
            result['chunk_size'] = len(doc_chunks)
            chunks.extend(doc_chunks)

        self.faiss = FAISS.from_documents(
            chunks,
            embedding=self.embedding_model
        )
        return result

    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform similarity search in the vector database.

        Args:
            query: The search query text
            k: Number of results to return
            filter: Optional metadata filter for the search

        Returns:
            List of document chunks most similar to the query

        Raises:
            ValueError: If vector store has not been initialized
        """
        if not self.faiss:
            raise ValueError("Vector store has not been initialized with documents")

        return self.faiss.similarity_search(query=query, k=k, filter=filter)

    def reranking(self, query: str, docs: List[Document], top_n: int = 10) -> List[str]:
        """
        Rerank documents using Cohere's reranking model.

        Args:
            query: The search query text
            docs: List of documents to rerank
            top_n: Number of top results to return

        Returns:
            List of reranked document contents
        """
        doc_texts = [doc.page_content for doc in docs]
        rerank_response = self.cohere_client.rerank(
            model=RERANKER_MODEL,
            query=query,
            documents=doc_texts,
            top_n=top_n
        )
        return [docs[result.index].page_content for result in rerank_response.results]

    def get_relevant_docs(self, chromdb_query: str, rerank_query: str,
                         filter: Optional[Dict[str, Any]] = None,
                         chunk_size: int = VECTOSTORE_TOPK) -> List[str]:
        """
        Perform a two-stage retrieval: vector search followed by reranking.

        Args:
            chromdb_query: Query for the initial vector search
            rerank_query: Query for the reranking step (can be different)
            filter: Optional metadata filter for the search
            chunk_size: Number of chunks in the document(s)

        Returns:
            List of the most relevant document contents
        """
        # Calculate appropriate values for k in both retrieval stages
        dense_topk = min(chunk_size, VECTOSTORE_TOPK)
        reranking_topk = min(chunk_size, COHERERANK_TOPN)

        # First stage: vector search
        docs = self.similarity_search(chromdb_query, filter=filter, k=dense_topk)

        # Second stage: reranking (if we have results)
        if docs:
            return self.reranking(rerank_query, docs, top_n=reranking_topk)
        return []