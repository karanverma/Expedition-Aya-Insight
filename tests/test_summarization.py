import pytest
import pandas as pd
from langchain.schema import Document
from app.processor import summarize # Adjust this import path as needed

@pytest.fixture
def dummy_pages():
    return [
        Document(
            page_content="LangChain is a framework for developing LLM-based applications.",
            metadata={"source": "sample_paper.pdf", "page": 1},
        ),
        Document(
            page_content="It provides utilities for prompt management, chains, and agents.",
            metadata={"source": "sample_paper.pdf", "page": 2},
        ),
    ]

def test_summarize_documents_returns_dataframe(dummy_pages):
    summary_df = summarize_documents(dummy_pages)

    assert isinstance(summary_df, pd.DataFrame)
    assert "file_name" in summary_df.columns
    assert "page_number" in summary_df.columns
    assert "chunks" in summary_df.columns
    assert "concise_summary" in summary_df.columns
    assert len(summary_df) == len(dummy_pages)
    assert summary_df["page_number"].iloc[0] == 1
