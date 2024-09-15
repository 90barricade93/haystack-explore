# -*- coding: utf-8 -*-
"""
In this tutorial, we will see how we can embed metadata as well as the text of a document. 
We will fetch various pages from Wikipedia and index them into an InMemoryDocumentStore with metadata information that includes their title, and URL. 
"""

import wikipedia
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy

def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):
    pipeline = Pipeline()
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
        model="thenlper/gte-large", meta_fields_to_embed=metadata_fields_to_embed
    ))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    return pipeline

def fetch_wikipedia_docs(titles):
    return [Document(content=wikipedia.page(title=title, auto_suggest=False).content,
                     meta={"title": title, "url": wikipedia.page(title=title, auto_suggest=False).url})
            for title in titles]

def create_retrieval_pipeline(document_store, document_store_with_metadata):
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large"))
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, scale_score=False, top_k=3))
    pipeline.add_component("retriever_with_embeddings", 
                           InMemoryEmbeddingRetriever(document_store=document_store_with_metadata, scale_score=False, top_k=3))

    pipeline.connect("text_embedder", "retriever")
    pipeline.connect("text_embedder", "retriever_with_embeddings")

    return pipeline

def main():
    some_bands = ["The Beatles", "The Cure"]
    raw_docs = fetch_wikipedia_docs(some_bands)

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    document_store_with_embedded_metadata = InMemoryDocumentStore(embedding_similarity_function="cosine")

    indexing_pipeline = create_indexing_pipeline(document_store=document_store)
    indexing_with_metadata_pipeline = create_indexing_pipeline(
        document_store=document_store_with_embedded_metadata, metadata_fields_to_embed=["title"]
    )

    indexing_pipeline.run({"cleaner": {"documents": raw_docs}})
    indexing_with_metadata_pipeline.run({"cleaner": {"documents": raw_docs}})

    retrieval_pipeline = create_retrieval_pipeline(document_store, document_store_with_embedded_metadata)

    result = retrieval_pipeline.run({"text_embedder": {"text": "Have the Beatles ever been to Bangor?"}})

    print("Retriever Results:\n")
    for doc in result["retriever"]["documents"]:
        print(doc)

    print("\nRetriever with Embeddings Results:\n")
    for doc in result["retriever_with_embeddings"]["documents"]:
        print(doc)

if __name__ == "__main__":
    main()
