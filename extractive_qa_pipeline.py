'''
What is extractive question answering? So glad you asked! 
The short answer is that extractive models pull verbatim answers out of text. 
It's good for use cases where accuracy is paramount, and you need to know exactly where in the text that the answer came from. 
If you want additional context, here's a deep dive on extractive versus generative language models. 
'''

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.telemetry import tutorial_running

def enable_telemetry():
    tutorial_running(34)

def load_and_prepare_data():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    return [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

def create_document_store():
    return InMemoryDocumentStore()

def create_indexing_pipeline(document_store, model):
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    return indexing_pipeline

def create_extractive_qa_pipeline(document_store, model):
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader()
    reader.warm_up()

    qa_pipeline = Pipeline()
    qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
    qa_pipeline.add_component(instance=retriever, name="retriever")
    qa_pipeline.add_component(instance=reader, name="reader")
    qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    qa_pipeline.connect("retriever.documents", "reader.documents")
    return qa_pipeline

def run_query(pipeline, query, top_k_retriever=3, top_k_reader=2):
    return pipeline.run(
        data={
            "embedder": {"text": query},
            "retriever": {"top_k": top_k_retriever},
            "reader": {"query": query, "top_k": top_k_reader}
        }
    )

def main():
    enable_telemetry()

    model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    documents = load_and_prepare_data()
    document_store = create_document_store()

    # Indexing pipeline
    indexing_pipeline = create_indexing_pipeline(document_store, model)
    indexing_pipeline.run({"documents": documents})

    # Extractive QA pipeline
    qa_pipeline = create_extractive_qa_pipeline(document_store, model)

    # Run a query
    query = "Who was Pliny the Elder?"
    result = run_query(qa_pipeline, query)
    print(result)

if __name__ == "__main__":
    main()
