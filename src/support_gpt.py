import os

from config.config import api_key, chat_history_path, persist_directory, glob

import faiss

from datetime import datetime

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] = api_key


class SupportGPT(object):
    def __init__(self):
        self.path = chat_history_path
        self.persist_directory = persist_directory
        self.glob = glob
        self.chat_history = list()
        self._DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does not know.

        Relevant pieces of previous conversation:
        {history}

        (You do not need to use these pieces of information if not relevant)

        Current conversation:
        Human: {input}
        AI:"""

    def generate_docs(self):
        loader = DirectoryLoader(path=self.path, glob=self.glob, loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    def generate_chroma_memory(self):
        docs = self.generate_docs()
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(persist_directory=self.persist_directory,
                                             documents=docs, embedding=embeddings)
        retriever = vector_store.as_retriever()
        return VectorStoreRetrieverMemory(retriever=retriever)

    @staticmethod
    def generate_faiss_memory():
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        vector_store = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vector_store.as_retriever()
        return VectorStoreRetrieverMemory(retriever=retriever)

    def generate_conversation_chain(self):
        llm = OpenAI(temperature=0, max_tokens=256)
        prompt = PromptTemplate(
            input_variables=["history", "input"], template=self._DEFAULT_TEMPLATE
        )
        conversation_with_summary = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=self.generate_faiss_memory() if len(os.listdir(self.path)) == 0 else self.generate_chroma_memory()
        )

        return conversation_with_summary

    def save_history(self):
        with open(self.path + f'{datetime.now().date()}-{datetime.now().hour}-'
                              f'{datetime.now().minute}-{datetime.now().second}.txt', 'a') as f:
            f.write(''.join(self.chat_history))

    def run(self):
        conversation_with_summary = self.generate_conversation_chain()

        while True:
            prompt = input('> ')
            if prompt == 'stop':
                if len(self.chat_history) == 0:
                    self.chat_history.append('Human: ' + '\n' + 'AI: ' + '\n')
                break
            output = conversation_with_summary.predict(input=prompt)
            print('>> ' + output)
            self.chat_history.append('Human: ' + prompt + '\n' + 'AI: ' + output + '\n')
