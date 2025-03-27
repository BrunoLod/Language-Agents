from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.schema import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Naive example for a conversational RAG

class ConversationalRag:
    """
    Conversational RAG (Retrieval-Augmented Generation) system for handling 
    conversational queries with context-aware retrieval.
    """
    def __init__(
            self, 
            llm: BaseChatModel, 
            system_message: str, 
            contextualize_message: str,
            embedding: Embeddings, 
            chat_history: ChatMessageHistory,
            documents: List[Document]
        ) -> None:
        """
        Initializes the ConversationalRag instance.

        Args:
            llm (BaseChatModel): The language model used for response generation.
            system_message (str): The system-level instruction message.
            contextualize_message (str): The message to provide context-aware queries.
            embedding (Embeddings): The embedding model used for document retrieval.
            chat_history (ChatMessageHistory): The chat history manager.
            documents (List[Document]): A list of documents to be processed and retrieved.
        """
        self.llm                   = llm 
        self.system_message        = system_message
        self.contextualize_message = contextualize_message
        self.embedding             = embedding
        self.chat_history          = chat_history
        self.documents             = documents
        self.store                 = {}
        self.__format_prompt()

    def loader(self) -> List[Document]:
        """
        Loads the documents from the provided sources using a web-based loader.

        Returns:
            List[Document]: A list of loaded documents.
        """
        loader = PyPDFLoader(self.documents)
        return loader.load()
    
    def splitter(self) -> List[Document]:
        """
        Splits the loaded documents into smaller chunks using a recursive text splitter.

        Returns:
            List[Document]: A list of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size         = 500, 
            chunk_overlap      = 50, 
            length_function    = len,
            separators         = ["", " ", ".", "\n", "\n\n"],
            is_separator_regex = False
        )
        return text_splitter.split_documents(self.loader())
    
    def retriever(self) -> InMemoryVectorStore: 
        """
        Converts the split documents into a vector store and returns a retriever for querying.

        Returns:
            InMemoryVectorStore: A retriever object for querying the vector store.
        """
        vector_store = InMemoryVectorStore.from_documents(
            self.splitter(), 
            self.embedding
        )
        return vector_store.as_retriever()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory: 
        """ 
        Retrieves or initializes the chat history for a given session.

        Args:
            session_id (str): The unique identifier for the chat session.
        
        Returns:
            BaseChatMessageHistory: The chat history associated with the session.
        """ 
        if session_id not in self.store: 
            self.store[session_id] = self.chat_history
        return self.store[session_id]

    def __format_prompt(self) -> None: 
        """ 
        Formats the system and contextualization prompts for structured conversation handling.
        """ 
        self.__contextualize_prompt = ChatPromptTemplate(
            [
                ("system", self.contextualize_message), 
                MessagesPlaceholder("chat_history"), 
                ("human", "{input}") 
            ]
        )

        self.__system_prompt = ChatPromptTemplate(
            [
                ("system", self.system_message),
                MessagesPlaceholder("chat_history"), 
                ("human", "{input}")
            ]
        )

    def buid_conversational_chain(self) -> Runnable:
        """ 
        Builds the conversational RAG chain by combining retrieval and response generation.

        Returns:
            Runnable: A runnable chain for processing conversational queries.
        """ 
        history_aware_retriever = create_history_aware_retriever(
            self.llm, 
            self.retriever(), 
            self.__contextualize_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm, 
            self.__system_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever, 
            question_answer_chain
        )

        return rag_chain

    def run(self, query: str) -> str:
        """ 
        Executes the RAG pipeline for a given query and returns the generated response.

        Args:
            query (str): The user input query.
        
        Returns:
            str: The generated response from the conversational model.
        """ 
        conversational_rag_chain = RunnableWithMessageHistory(
            self.buid_conversational_chain(), 
            self.get_session_history, 
            input_messages_key   = "input", 
            history_messages_key = "chat_history", 
            output_messages_key  = "answer"
        )

        response = conversational_rag_chain.invoke(
            {"input": query}, 
            config={
                "configurable": {"session_id": 935}
            }
        )["answer"]

        return response
    
if __name__=="__main__": 
    
    from chatbot_prompt.contextualize_message import contextualize_message
    from chatbot_prompt.system_message import system_message
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_groq import ChatGroq
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    llm = ChatGroq(
        model = "llama3-70b-8192", 
        temperature = 0.5, 
        api_key = "your-api-key"  
    )

    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2"
    )

    chat_history = ChatMessageHistory()

    article = "../data/Int J Mental Health Nurs - 2023 - Higgins - Artificial intelligence  AI  and machine learning  ML  based decision support.pdf"

    conversational_rag = ConversationalRag(
        llm                   = llm, 
        system_message        = system_message, 
        contextualize_message = contextualize_message,
        embedding             = embedding, 
        chat_history          = chat_history,
        documents             = article
    )
    
    print("Olá! Eu sou a Lily, prazer. O que deseja conversar, caro(a) morceguinho(a) ?")
    while True: 

        user_input = input("Palavras soltas: ")
        if user_input.lower() in ["sair", "exit"]: 
            print("Até mais! Trombamos por aí depois...")
            break
        try:
            response = conversational_rag.run(query=user_input)
            print(f"Lily: {response}")
        except Exception as e: 
            print(f"Erro ao processar a mensagem {e}")
