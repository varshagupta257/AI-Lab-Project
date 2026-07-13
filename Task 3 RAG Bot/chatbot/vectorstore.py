import re
import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration matching your successful offline setup
FORCED_SECTIONS_BY_CATEGORY = {
    "Entertainment":   ["7.1", "9.7"],
    "Meals":           ["7.1"],
    "Travel":          ["7.3"],
    "Office Supplies": ["7.5", "9.1"],
    "Software":        ["7.4"],
    "Miscellaneous":   ["7.6", "9.3", "9.4"],
    "Wi-Fi":           ["7.7"],
}
ALWAYS_FORCED = ["Appendix E", "Appendix F", "Appendix G", "6."]

SECTION_HEADER_PATTERN = re.compile(
    r"(?:^|\n)((?:\d+(?:\.\d+)*\.?\s+[A-Z][^\n]*)|(?:Appendix [A-G][^\n]*))",
)
MAX_HEADER_LEN = 90

# Global instance to persist the retriever in memory during the session
_retriever_instance = None

class StructureAwareRetriever:
    def __init__(self):
        # Keeps the exact same embedding model you optimized
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.sections: Dict[str, str] = {}

    def index_policy(self, policy_text: str):
        raw_matches = list(SECTION_HEADER_PATTERN.finditer(policy_text))
        valid = []
        
        for m in raw_matches:
            raw_title = m.group(1).strip()
            if re.search(r"\.{3,}", raw_title) or re.search(r"\.{3,}", policy_text[m.end():m.end() + 40]):
                continue
            if len(raw_title) > MAX_HEADER_LEN:
                continue
            title = re.sub(r"\s+\d+\s*$", "", raw_title)
            valid.append((m.start(), m.end(), title))

        for i, (h_start, h_end, title) in enumerate(valid):
            end = valid[i + 1][0] if i + 1 < len(valid) else len(policy_text)
            body = policy_text[h_end:end].strip()
            if body and title not in self.sections:
                self.sections[title] = body

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        child_docs = []
        for title, sec_body in self.sections.items():
            for chunk in child_splitter.split_text(sec_body):
                child_docs.append(Document(page_content=f"{title}\n{chunk}", metadata={"section": title}))
        self.vectorstore = FAISS.from_documents(child_docs, self.embeddings)

    def semantic_context(self, query: str, k: int = 2) -> List[str]:
        if not self.vectorstore:
            return []
        matches = self.vectorstore.similarity_search(query, k=k)
        seen = []
        for doc in matches:
            sec = doc.metadata.get("section")
            if sec and sec not in seen:
                seen.append(sec)
        return seen

    def forced_titles(self, prefixes: List[str]) -> List[str]:
        found = []
        for prefix in prefixes:
            match = next((full for full in self.sections if full.startswith(prefix)), None)
            if match and match not in found:
                found.append(match)
        return found

    def render(self, titles: List[str]) -> str:
        return "\n\n---\n\n".join(f"[{t}]\n{self.sections[t]}" for t in titles if t in self.sections)


def get_or_create_retriever(policy_text: str) -> StructureAwareRetriever:
    """Helper to initialize and cache the retriever instance so we don't re-index on every message."""
    global _retriever_instance
    if _retriever_instance is None and policy_text.strip():
        print("Building structure-aware retriever...")
        retriever = StructureAwareRetriever()
        retriever.index_policy(policy_text)
        _retriever_instance = retriever
        print(f"Indexed {len(retriever.sections)} policy sections.")
    return _retriever_instance


def query_structure_aware_rag(query: str, category: str = "Meals", level: str = "Junior") -> str:
    """
    Executes the exact hybrid retrieval logic:
    Combines ALWAYS_FORCED + Category-Specific + Semantic Matches.
    """
    global _retriever_instance
    if _retriever_instance is None:
        return "No policy text indexed yet."

    retriever = _retriever_instance
    
    # 1. Grab always forced sections (Appendix E, F, G, Section 6)
    always = retriever.forced_titles(ALWAYS_FORCED)
    
    # 2. Grab category specific rules (e.g., Section 7.1 for Meals)
    category_specific = retriever.forced_titles(FORCED_SECTIONS_BY_CATEGORY.get(category, []))
    
    # 3. Perform the exact semantic retrieval search (k=2)
    semantic_query = f"{category} claim by {level}: {query}"
    semantic = retriever.semantic_context(semantic_query, k=2)

    # 4. Combine and deduplicate
    titles = list(dict.fromkeys(always + category_specific + semantic))
    
    # 5. Render sections back as context string
    return retriever.render(titles)