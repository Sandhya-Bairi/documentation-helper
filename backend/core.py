import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain import hub
from consts import INDEX_NAME

# pc = Pinecone


def run_llm(query: str):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    question_answer_chain = create_stuff_documents_chain(llm=chat, prompt=retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(retriever=docsearch.as_retriever(), combine_docs_chain=question_answer_chain)

    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )
    #return qa({"query": query})
    return qa.invoke({"input": query})


if __name__ == "__main__":
    print(run_llm(query="What is RetrievalQA Chain?"))
