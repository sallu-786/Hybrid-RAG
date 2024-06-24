
from embeddings import get_file, get_text_chunks, create_embeddings
import streamlit as st

# when file is uploaded by user, create new vector data for that file
def create_new_vector_db(file):
    with st.spinner("Creating vector data"):
        text = get_file(file)
        text_chunks = get_text_chunks(text)
        # print(text_chunks)
        # input("chunks-----------")
        vectordb = create_embeddings(text_chunks)
        # print(vectordb)
        # input("vector-----------")
    return vectordb,text_chunks

def handle_file_upload(file):
    if file:
        vectordb,text_chunks = create_new_vector_db(file)
        st.write("Vector data created successfully.")
        return vectordb,text_chunks

    else:                             
        pass

def normalize_scores(results, score_type):
    """
    Normalizes scores based on the type of scoring mechanism.
    For BM25 (higher is better), scores are normalized as-is.
    For FAISS similarity (lower is better), scores are inverted.
    """
    scores = [score for (_, score) in results]
    if score_type == 'bm25':
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        normalized_scores = [(score - min_score) / (max_score - min_score + 1e-10) for score in scores]
    elif score_type == 'faiss':
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        normalized_scores = [(max_score - score) / (max_score - min_score + 1e-10) for score in scores]
    return normalized_scores

def add_scores(results, weight, normalized_scores, merged_scores):
    """
    Adds weighted scores to the merged_scores dictionary.
    """
    for rank, ((doc, _), normalized_score) in enumerate(zip(results, normalized_scores)):
        if isinstance(doc, dict):
            doc_id = (doc['metadata']['page'], doc['content'])
        else:
            doc_id = (doc.metadata['page'], doc.page_content)

        if doc_id in merged_scores:
            merged_scores[doc_id] += weight * normalized_score / (rank + 1)
        else:
            merged_scores[doc_id] = weight * normalized_score / (rank + 1)

def rrf(bm25_results, vector_results, k=3):
    """
    Combines BM25 and vector search results using Reciprocal Rank Fusion.
    """
    merged_scores = {}

    # bm25_normalized_scores = normalize_scores(bm25_results, 'bm25')
    bm25_normalized_scores = normalize_scores(bm25_results, 'bm25')
  
    vector_normalized_scores = normalize_scores(vector_results, 'faiss')

    add_scores(bm25_results, weight=0.5, normalized_scores=bm25_normalized_scores, merged_scores=merged_scores)
    add_scores(vector_results, weight=0.5, normalized_scores=vector_normalized_scores, merged_scores=merged_scores)

    sorted_docs = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
    final_results = [{"content": content, "metadata": {"page_number": page_number}} for (page_number, content), score in sorted_docs[:k]]
    return final_results
