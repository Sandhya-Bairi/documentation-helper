from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone

from consts import INDEX_NAME

load_dotenv()


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/api.python.langchain.com/en/latest/chains",
        encoding="utf-8",
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")

    embeddings = OpenAIEmbeddings()
    PineconeLangChain.from_documents(
        documents=documents, embedding=embeddings, index_name=INDEX_NAME
    )
    print("******** Added to Pinecone vector store ********")


if __name__ == "__main__":
    ingest_docs()
