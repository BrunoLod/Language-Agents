from typing import List

from langchain.schema import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters.base import TextSplitter


class RagAgent():
    """ 
    """
    def __init__(
            self, 
            llm: BaseChatModel, 
            system_prompt: str,
            emebedding: Embeddings, 
            retriever: VectorStore,
            loader: BaseLoader,
            splitter: TextSplitter            
        ) -> None:
        
        self.llm           = llm
        self.system_prompt = system_prompt
        self.embedding     = emebedding
        self.retriever     = retriever
        self.loader        = loader
        self.splitter      = splitter

    def loader(self) -> List[Document]:
        """ 
        """
        return self.loader.load()
    
    def splitter(self) -> List[Document]:
        """ 
        """
        text_splitter = self.splitter
        return text_splitter.split_documents(self.loader())
    
    def retriever(self) -> VectorStore: 
        """ 
        """

        vector_store = self.retriever.from_documents(
            self.splitter(), 
            self.embedding
        )
        return vector_store.as_retriever()
    
    def run(self, query: str) -> str: 
        """ 
        """
        rag_chain = (
            {"context": self.retriever(), "question": RunnablePassthrough()}
            | self.system_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(query)
    
if __name__=="__main__": #pragma: no-cover

    from chatbot_prompt.rag_agent_prompt import system_rag_prompt
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_groq import ChatGroq
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    llm = ChatGroq(
        model = "llama3-70b-8192", 
        temperature = 0.5, 
        api_key = "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"  
    )

    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size         = 500, 
        chunk_overlap      = 50, 
        length_function    = len,
        separators         = ["", " ", ".", "\n", "\n\n"],
        is_separator_regex = False
    )

    url = "https://en-m-wikipedia-org.translate.goog/wiki/Dark_wave?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=tc"

    loader = WebBaseLoader(url)

    rag_agent = RagAgent(
        llm           = llm, 
        system_prompt = system_rag_prompt, 
        embedding     = embedding, 
        retriever     = InMemoryVectorStore, 
        loader        = loader, 
        splitter      = text_splitter
    )

    print("Olá! Eu sou a Lily, prazer. O que deseja conversar, caro(a) morceguinho(a) ?")
    while True: 

        user_input = input("Palavras soltas: ")
        if user_input.lower() in ["sair", "exit"]: 
            print("Até mais! Trombamos por aí depois...")
            break
        try:
            response = rag_agent.run(query=user_input)
            print(f"Lily: {response}")
        except Exception as e: 
            print(f"Erro ao processar a mensagem {e}")


