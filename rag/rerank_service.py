from http import HTTPStatus
from typing import List

from dashscope import TextReRank
from langchain_core.documents import Document

from utils.config_handler import rag_conf, chroma_conf
from utils.logger_handler import logger


class RerankService:
    def __init__(self):
        self.model_name = rag_conf.get("rerank_model_name", "gte-rerank-v2")
        self.top_k = int(chroma_conf.get("k", 3))

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        try:
            doc_texts = [doc.page_content for doc in docs]
            response = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=doc_texts,
                top_n=self.top_k,
                return_documents=False,
            )
            if response.status_code != HTTPStatus.OK or not response.output:
                logger.warning(
                    f"[Rerank]调用失败，status_code={response.status_code}, code={response.code}, message={response.message}"
                )
                return docs[: self.top_k]

            result_docs: List[Document] = []
            for item in response.output.results or []:
                idx = int(item.index)
                if 0 <= idx < len(docs):
                    result_docs.append(docs[idx])

            return result_docs if result_docs else docs[: self.top_k]
        except Exception as e:
            logger.warning(f"[Rerank]重排序异常，降级为向量检索结果: {str(e)}")
            return docs[: self.top_k]
