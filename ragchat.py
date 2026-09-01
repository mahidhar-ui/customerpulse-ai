from langchain_community.vectorstores import Chroma

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain.chains import (
    RetrievalQA
)


embeddings = OpenAIEmbeddings()


vectorstore = Chroma(
    persist_directory="rag/vector_db",
    embedding_function=embeddings
)


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


def ask_question(question):

    result = qa_chain.invoke(
        {
            "query": question
        }
    )

    return result