'''
Although new retrieval techniques are great, 
sometimes you just know that you want to perform search on a specific group of documents in your document store. 
This can be anything from all the documents that are related to a specific user, or that were published after a certain date and so on. 
Metadata filtering is very useful in these situations.
'''

from datetime import datetime
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.telemetry import tutorial_running

def enable_telemetry():
    tutorial_running(31)

def create_documents():
    return [
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
            meta={"version": 1.15, "date": datetime(2023, 3, 30)},
        ),
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack[inference]. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
            meta={"version": 1.22, "date": datetime(2023, 11, 7)},
        ),
        Document(
            content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai. The haystack-ai package is built on the main branch which is an unstable beta version, but it's useful if you want to try the new features as soon as they are merged.",
            meta={"version": 2.0, "date": datetime(2023, 12, 4)},
        ),
    ]

def initialize_document_store(documents):
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
    document_store.write_documents(documents=documents)
    return document_store

def create_pipeline(document_store):
    pipeline = Pipeline()
    pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
    return pipeline

def run_simple_query(pipeline, query, version_filter):
    return pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {"field": "meta.version", "operator": ">", "value": version_filter}
            }
        }
    )

def run_complex_query(pipeline, query, version_filter, date_filter):
    return pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.version", "operator": ">", "value": version_filter},
                        {"field": "meta.date", "operator": ">", "value": date_filter},
                    ],
                },
            }
        }
    )

def main():
    enable_telemetry()
    
    documents = create_documents()
    document_store = initialize_document_store(documents)
    pipeline = create_pipeline(document_store)
    
    query = "Haystack installation"
    
    print("Simple query results:")
    simple_results = run_simple_query(pipeline, query, 1.21)
    print(simple_results)
    
    print("\nComplex query results:")
    complex_results = run_complex_query(pipeline, query, 1.21, datetime(2023, 11, 7))
    print(complex_results)

if __name__ == "__main__":
    main()
