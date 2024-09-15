'''
In this tutorial, you’ll build an indexing pipeline that preprocesses different types of files (markdown, txt and pdf). 
Each file will have its own FileConverter. 
The rest of the indexing pipeline is fairly standard - split the documents into chunks, trim whitespace, create embeddings and write them to a Document Store.
'''

import os
import gdown
from pathlib import Path
from getpass import getpass

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator

def download_files(url, output_dir):
    gdown.download_folder(url, quiet=True, output=output_dir)

def create_indexing_pipeline(document_store):
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()
    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    document_writer = DocumentWriter(document_store)

    pipeline = Pipeline()
    pipeline.add_component(instance=file_type_router, name="file_type_router")
    pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    pipeline.add_component(instance=document_joiner, name="document_joiner")
    pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    pipeline.add_component(instance=document_splitter, name="document_splitter")
    pipeline.add_component(instance=document_embedder, name="document_embedder")
    pipeline.add_component(instance=document_writer, name="document_writer")

    pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    pipeline.connect("text_file_converter", "document_joiner")
    pipeline.connect("pypdf_converter", "document_joiner")
    pipeline.connect("markdown_converter", "document_joiner")
    pipeline.connect("document_joiner", "document_cleaner")
    pipeline.connect("document_cleaner", "document_splitter")
    pipeline.connect("document_splitter", "document_embedder")
    pipeline.connect("document_embedder", "document_writer")

    return pipeline

def create_query_pipeline(document_store):
    template = """
    Answer the questions based on the given context.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ question }}
    Answer:
    """
    
    pipe = Pipeline()
    pipe.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component(
        "llm",
        HuggingFaceAPIGenerator(api_type="serverless_inference_api", api_params={"model": "HuggingFaceH4/zephyr-7b-beta"}),
    )

    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")

    return pipe

def main():
    # Download files
    url = "https://drive.google.com/drive/folders/1n9yqq5Gl_HWfND5bTlrCwAOycMDt5EMj"
    output_dir = "recipe_files"
    download_files(url, output_dir)

    # Create document store
    document_store = InMemoryDocumentStore()

    # Create and run indexing pipeline
    indexing_pipeline = create_indexing_pipeline(document_store)
    indexing_pipeline.run({"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}})

    # Create query pipeline
    query_pipeline = create_query_pipeline(document_store)

    # Run query
    question = "What ingredients would I need to make vegan keto eggplant lasagna, vegan persimmon flan, and vegan hemp cheese?"
    result = query_pipeline.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"question": question},
            "llm": {"generation_kwargs": {"max_new_tokens": 350}},
        }
    )
    print(result)

if __name__ == "__main__":
    # Set up Hugging Face API token
    if "HF_API_TOKEN" not in os.environ:
        os.environ["HF_API_TOKEN"] = getpass("Enter Hugging Face token:")

    main()
