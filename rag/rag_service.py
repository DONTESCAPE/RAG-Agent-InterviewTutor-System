"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.query_optimizer import QueryOptimizer
from rag.rerank_service import RerankService
from rag.self_rag import SelfRAGRouter
from utils.config_handler import chroma_conf, rag_conf
from utils.prompt_loader import load_rag_prompts
from rag.vector_store import VectorStoreService


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.enable_rerank = bool(rag_conf.get("enable_rerank", False))
        self.recall_k = int(rag_conf.get("rerank_recall_k", 12))
        self.final_k = int(chroma_conf.get("k", 3))
        self.query_optimizer = QueryOptimizer()
        self.self_rag_router = SelfRAGRouter()
        retriever_k = self.recall_k if self.enable_rerank else self.final_k
        self.retriever = self.vector_store.get_retriever(k=retriever_k)
        self.rerank_service = RerankService() if self.enable_rerank else None
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def _retrieve_with_multi_recall(self, query: str, history=None) -> list[Document]:
        recall_queries = self.query_optimizer.build_recall_queries(query, history)
        docs_groups = []
        for recall_query in recall_queries:
            try:
                docs = self.retriever.invoke(recall_query)
                docs_groups.append(docs)
            except Exception:
                continue
        return self.query_optimizer.merge_documents(docs_groups)

    def retriever_docs(self, query: str, history=None) -> list[Document]:
        optimized_query = self.query_optimizer.get_rerank_query(query, history)
        docs = self._retrieve_with_multi_recall(query, history)
        if not docs:
            try:
                docs = self.retriever.invoke(optimized_query)
            except Exception:
                docs = []

        if self.self_rag_router.enabled:
            decision = self.self_rag_router.should_route(optimized_query, docs)
            if decision.should_self_reflect:
                return docs[: self.final_k]

        if not self.enable_rerank or not self.rerank_service:
            return docs[: self.final_k]
        return self.rerank_service.rerank(optimized_query, docs)

    def _build_context(self, query: str, history=None) -> str:
        context_docs = self.retriever_docs(query, history)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        return context

    def rag_summarize(self, query: str, history=None) -> str:
        context = self._build_context(query, history)
        return self.chain.invoke({"input": query, "context": context})

    def rag_summarize_stream(self, query: str, history=None) -> Iterator[str]:
        context = self._build_context(query, history)
        for chunk in self.chain.stream({"input": query, "context": context}):
            if chunk:
                yield str(chunk)


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("什么是线程？"))
