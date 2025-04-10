from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_cohere.llms import Cohere
import pandas as pd
from pathlib import Path as p

def summarize_documents(pages):
    llm = Cohere(model="command-a-03-2025")

    map_prompt = PromptTemplate(
        template="""
        Write a summary of this chunk of text that includes the main points and any important details.
        {text}
        """,
        input_variables=["text"],
    )

    combine_prompt = PromptTemplate(
        template="""
        Write a concise summary of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
        """,
        input_variables=["text"],
    )

    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
    )

    map_reduce_outputs = map_reduce_chain({"input_documents": pages})

    final_mp_data = []
    for doc, out in zip(
        map_reduce_outputs["input_documents"], map_reduce_outputs["intermediate_steps"]
    ):
        metadata = doc.metadata
        output = {
            "file_name": p(metadata.get("source", "unknown")).stem,
            "file_type": p(metadata.get("source", "unknown")).suffix,
            "page_number": metadata.get("page", -1),
            "chunks": doc.page_content,
            "concise_summary": out,
        }
        final_mp_data.append(output)

    pdf_mp_summary = pd.DataFrame.from_dict(final_mp_data)
    pdf_mp_summary = pdf_mp_summary.sort_values(
        by=["file_name", "page_number"]
    ).reset_index(drop=True)

    return pdf_mp_summary

# Example usage:
def print_summary_for_index(df, index=3):
    print("[Context]")
    print(df["chunks"].iloc[index])
    print("\n\n[Simple Summary]")
    print(df["concise_summary"].iloc[index])
    print("\n\n[Page number]")
    print(df["page_number"].iloc[index])
    print("\n\n[Source: file_name]")
    print(df["file_name"].iloc[index])
