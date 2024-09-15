'''
The process involves four main components: 
SentenceTransformersTextEmbedder for creating an embedding for the user query, 
InMemoryBM25Retriever for fetching relevant documents, PromptBuilder for creating a template prompt, 
and OpenAIGenerator for generating responses.
'''

import os
from getpass import getpass
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

def initialize_document_store():
    return InMemoryDocumentStore()

def fetch_and_prepare_data():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    return [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

def create_document_embedder():
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    embedder.warm_up()
    return embedder

def index_documents(document_store, docs, doc_embedder):
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

def create_rag_pipeline(document_store):
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = InMemoryEmbeddingRetriever(document_store)
    
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """
    prompt_builder = PromptBuilder(template=template)
    
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
    generator = OpenAIGenerator(model="gpt-3.5-turbo")
    
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", generator)
    
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    
    return pipeline

def ask_question(pipeline, question):
    response = pipeline.run({
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question}
    })
    return response["llm"]["replies"][0]

def main():
    document_store = initialize_document_store()
    docs = fetch_and_prepare_data()
    doc_embedder = create_document_embedder()
    index_documents(document_store, docs, doc_embedder)
    
    rag_pipeline = create_rag_pipeline(document_store)
    
    question = "Where is Gardens of Babylon?"
    answer = ask_question(rag_pipeline, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Additional example questions
    examples = [
        "Why did people build Great Pyramid of Giza?",
        "What does Rhodes Statue look like?",
        "Why did people visit the Temple of Artemis?",
        "What is the importance of Colossus of Rhodes?",
        "What happened to the Tomb of Mausolus?",
        "How did Colossus of Rhodes collapse?",
    ]
    
    for question in examples:
        print(f"\nQuestion: {question}")
        print(f"Answer: {ask_question(rag_pipeline, question)}")

if __name__ == "__main__":
    main()
