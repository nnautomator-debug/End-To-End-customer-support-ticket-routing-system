import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

load_dotenv(os.path.join(ROOT, ".env"))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "end-to-end-nlp-project")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA_PATH = os.path.join(ROOT, "Data", "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
INDEX_DIR = os.path.join(HERE, "faiss_index")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EN_LANG = "en"
SAMPLE_SIZE = 10_000
TOP_K = 5
GROQ_MODEL = "openai/gpt-oss-20b"

META_COLS = (
    "subject",
    "body",
    "type",
    "queue",
    "priority",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
    "tag_5",
)


class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL, **kwargs: object):
        self._model = SentenceTransformer(model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.encode(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()


_EMBEDDINGS: MiniLMEmbeddings | None = None
_VECTORSTORE: FAISS | None = None


def get_embeddings() -> MiniLMEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = MiniLMEmbeddings()
    return _EMBEDDINGS


def get_vectorstore() -> FAISS:
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = FAISS.load_local(
            INDEX_DIR,
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _VECTORSTORE


def load_english_answers(limit: int | None = None) -> list[Document]:
    df = pd.read_csv(DATA_PATH)
    df = df[df["language"].astype(str).str.strip().str.lower() == EN_LANG]
    df = df.dropna(subset=["answer"])
    if limit is not None:
        df = df.head(limit)

    docs = []
    for idx, row in df.iterrows():
        meta = {"row_id": int(idx), "language": EN_LANG}
        for col in META_COLS:
            val = row[col]
            meta[col] = "" if pd.isna(val) else str(val)
        docs.append(Document(page_content=str(row["answer"]), metadata=meta))
    return docs


def build_faiss_index(docs: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(docs, embeddings)


def retrieve_similar_answers(ticket_body: str, k: int = TOP_K, vectorstore: FAISS | None = None) -> list[tuple[Document, float]]:
    vectorstore = vectorstore or get_vectorstore()

    query_vec = np.asarray(get_embeddings().embed_query(ticket_body), dtype="float32")
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    vectors = np.empty((vectorstore.index.ntotal, EMBEDDING_DIM), dtype="float32")
    for i in range(vectorstore.index.ntotal):
        vectors[i] = vectorstore.index.reconstruct(i)
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    similarities = (vectors / norms[:, None]) @ query_vec

    top_idx = np.argsort(-similarities)[:k]

    results = []
    for i in top_idx:
        doc_id = vectorstore.index_to_docstore_id[int(i)]
        doc = vectorstore.docstore.search(doc_id)
        results.append((doc, float(similarities[i])))
    return results


def ensure_index() -> FAISS:
    if not (os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) and os.path.exists(os.path.join(INDEX_DIR, "index.pkl"))):
        print(f"Index not found at {INDEX_DIR} -> building it first ...")
        docs = load_english_answers(limit=SAMPLE_SIZE)
        embeddings = get_embeddings()
        vectorstore = build_faiss_index(docs, embeddings)
        os.makedirs(INDEX_DIR, exist_ok=True)
        vectorstore.save_local(INDEX_DIR)
        print(f"Saved {vectorstore.index.ntotal} vectors -> {INDEX_DIR}")
    return get_vectorstore()


def build_rag_chain():
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a support assistant answering a customer ticket.\n"
                    "You are given the top 5 historical agent answers retrieved by cosine similarity.\n"
                    "Use ALL 5 chunks to draft the best possible answer for the customer.\n"
                    "Keep it short and concise.\n"
                    "Respond in bullet points only.\n"
                    "Do not invent facts that are not in the chunks."
                ),
            ),
            (
                "human",
                (
                    "Customer ticket:\n{ticket_body}\n\n"
                    "Retrieved agent answers:\n{context}"
                ),
            ),
        ]
    )

    return prompt | llm


def generate_answer(ticket_body: str, k: int = TOP_K, vectorstore: FAISS | None = None) -> str:
    results = retrieve_similar_answers(ticket_body, k=k, vectorstore=vectorstore)

    context_parts = []
    for rank, (doc, score) in enumerate(results, 1):
        context_parts.append(f"[{rank}] (similarity={score:.4f})\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    chain = build_rag_chain()
    response = chain.invoke({"ticket_body": ticket_body, "context": context})
    return response.content


def main():
    vectorstore = ensure_index()
    print(f"Loaded {vectorstore.index.ntotal} vectors from {INDEX_DIR}")

    sample_ticket = (
        "Dear Customer Support Team, I am urgently reporting a series of severe outages "
        "impacting several key devices critical to our operations. The affected devices "
        "include network switches, corporate laptops, and vital cloud-based software "
        "applications. The issues seem to originate from a potential infrastructure fault, "
        "and initial troubleshooting has involved hardware resets."
    )

    print(f"\nSample user ticket:\n{sample_ticket}\n")
    print(f"Retrieving top {TOP_K} similar historical agent answers by cosine similarity ...\n")

    results = retrieve_similar_answers(sample_ticket, k=TOP_K, vectorstore=vectorstore)

    for rank, (doc, score) in enumerate(results, 1):
        print(f"[#{rank}] cosine similarity = {score:.4f}")
        print(f"   subject : {doc.metadata.get('subject', '')}")
        print(f"   queue   : {doc.metadata.get('queue', '')}")
        print(f"   priority: {doc.metadata.get('priority', '')}")
        print(f"   answer  : {doc.page_content}\n")

    print("=" * 70)
    print("RAG answer from the LLM:")
    print("=" * 70)
    print(generate_answer(sample_ticket, k=TOP_K, vectorstore=vectorstore))


if __name__ == "__main__":
    main()