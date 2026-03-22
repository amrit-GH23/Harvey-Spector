"""
Jolly LLB — RAG Retrieval & Synthesis
======================================
1. Hybrid retrieval -> Top articles/sections via multi-collection search
2. Context injection -> Formatted source strings from Constitution, BNS, BNSS, BSA
3. System prompt -> Structured, readable legal responses
4. Output -> { answer, sources }
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import OLLAMA_BASE_URL, LLM_MODEL
from app.hybrid_search import hybrid_retrieve

SYSTEM_PROMPT = """You are **Jolly LLB**, a professional yet approachable AI Legal Assistant specializing in Indian Law.

You have expertise across FOUR pillars of Indian law:
- **Constitution of India** -- Fundamental Rights, governance, state policy
- **BNS** (Bharatiya Nyaya Sanhita) -- Criminal offences & punishments (replaced IPC)
- **BNSS** (Bharatiya Nagarik Suraksha Sanhita) -- Criminal procedure, arrest, bail (replaced CrPC)
- **BSA** (Bharatiya Sakshya Adhiniyam) -- Rules of evidence, digital proof (replaced Indian Evidence Act)

ANSWER FORMAT -- Always structure your response as follows:

**Applicable Law:**
State which specific Article(s) or Section(s) apply and from which law. Example: "Section 302 of BNS" or "Article 21 of the Constitution".

**In Simple Terms:**
Explain what the law means in plain, everyday language that any citizen can understand. Avoid legal jargon. Use relatable examples where helpful.

**Key Points:**
- Use bullet points to list the most important takeaways
- Include punishments, time limits, or conditions if applicable
- Mention related provisions that the user should also know about

**Important Note:** (only if needed)
Add any caveats, exceptions, or practical advice. For example: "This right is subject to reasonable restrictions under Article 19(2)" or "Always consult a practicing lawyer for case-specific advice."

RULES:
1. Use ONLY the provided Context to answer. Do not invent information.
2. Be specific -- cite exact Article/Section numbers with their title and source law.
3. Keep language conversational but accurate. You are a helpful legal friend, not a textbook.
4. If the query is clearly unrelated to law (e.g., cooking, sports, movies), politely say you are Jolly LLB, an Indian Law assistant, and suggest 2-3 legal questions instead.
5. If the query IS about law but the context doesn't fully cover it, answer with the closest relevant provisions and note that additional provisions may exist.
6. Use **bold** for emphasis on important terms, section numbers, and legal concepts.
7. When multiple laws apply to a query (e.g., a crime involves BNS for the offence, BNSS for procedure, and BSA for evidence), mention ALL relevant provisions across sources."""


def _get_llm() -> ChatOllama:
    """Fetch the LLM configuration fresh each time to avoid stale overrides."""
    from app.config import LLM_MODEL, OLLAMA_BASE_URL
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)


def _format_source(doc, idx: int) -> tuple[str, dict]:
    """Format a single document for context injection and source tracking."""
    source_type = doc.metadata.get("source_type", "constitution")

    if source_type == "constitution":
        art_no = doc.metadata.get("article_no", "?")
        title = doc.metadata.get("title", "")
        part = doc.metadata.get("part", "")
        label = f"Article {art_no} ({title}) [{part}]"
        source_info = {
            "article_no": art_no,
            "title": title,
            "part": part,
            "source_type": "constitution",
        }
    else:
        sec_no = doc.metadata.get("section_no", "?")
        title = doc.metadata.get("title", "")
        chapter = doc.metadata.get("chapter_name", "")
        law_label = source_type.upper()
        label = f"Section {sec_no} ({title}) [{law_label} -- Ch. {chapter}]"
        source_info = {
            "section_no": sec_no,
            "title": title,
            "chapter": chapter,
            "source_type": source_type,
        }

    context_str = f"Source {idx}: {label}\n{doc.page_content}"
    return context_str, source_info


def get_legal_advice(user_query: str) -> dict:
    """
    RAG workflow:
      1. Hybrid retrieve across Constitution + BNS + BNSS + BSA
      2. Format context with source labels
      3. Query LLM with structured legal system prompt
      4. Return { answer, sources }
    """
    # Step 1 -- Retrieve via multi-collection hybrid pipeline
    docs = hybrid_retrieve(user_query, final_k=5)

    if not docs:
        return {"answer": "No relevant articles or sections found.", "sources": []}

    # Step 2 -- Format context
    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        context_str, source_info = _format_source(doc, i)
        context_parts.append(context_str)
        sources.append(source_info)

    context = "\n\n---\n\n".join(context_parts)

    # Step 3 -- Query LLM (cached singleton)
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\n---\n\nUser Query: {user_query}"),
    ])

    # Step 4 -- Return
    return {"answer": response.content, "sources": sources}
