import os
import re
import csv
import time
import warnings
from collections import defaultdict
from typing import List, Dict, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# ==============================================================================
# CONFIGURATION
# ==============================================================================
POLICY_PDF_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1-merged.pdf"
TRANSACTIONS_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2withCAT.csv"
OUTPUT_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output_v8.csv"
OLLAMA_MODEL = "llama3.1"

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

# ==============================================================================
# PYTHON MATHEMATICAL PRE-PROCESSOR
# ==============================================================================
def parse_headcount_and_calc(justification: str, total_amount: str) -> tuple:
    """Extracts headcount via precise regex patterns and computes per-person costs cleanly in Python."""
    match = re.search(r'(?:group of|for)\s+(\d+)|(\d+)\s*(?:people|person|attendees|guests|team of)', justification, re.IGNORECASE)
    
    headcount = 1
    if match:
        headcount = int(match.group(1) or match.group(2))
    
    try:
        amount_val = float(str(total_amount).replace("$", "").strip())
    except ValueError:
        amount_val = 0.0

    per_person_cost = round(amount_val / headcount, 2) if headcount > 0 else amount_val
    return headcount, per_person_cost

# ==============================================================================
# STRUCTURE-AWARE RETRIEVER
# ==============================================================================
def load_policy_text() -> str:
    pdf_reader = PdfReader(POLICY_PDF_PATH)
    return "\n".join(p.extract_text() for p in pdf_reader.pages if p.extract_text())

class StructureAwareRetriever:
    def __init__(self):
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

    def semantic_context(self, query: str, k: int = 3) -> List[str]:
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

# ==============================================================================
# DATASET CO-OCCURRENCE SCANNER
# ==============================================================================
def find_cooccurrences(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    notes = defaultdict(list)

    same_charge = defaultdict(list)
    for r in rows:
        same_charge[(r["Merchant"], r["Amount"], r["Date"])].append(r["Transaction_ID"])
    for key, ids in same_charge.items():
        if len(ids) > 1:
            for tid in ids:
                others = [i for i in ids if i != tid]
                notes[tid].append(f"Same merchant, amount, and date also appears on: {others}")

    same_batch = defaultdict(list)
    for r in rows:
        same_batch[(r["Department"], r["Category"], r["Date"])].append(r)
    for key, group in same_batch.items():
        if len(group) > 1:
            total = sum(float(r["Amount"]) for r in group)
            for r in group:
                others = [g["Transaction_ID"] for g in group if g["Transaction_ID"] != r["Transaction_ID"]]
                notes[r["Transaction_ID"]].append(
                    f"Same department + category + date includes: {others} (Combined total: ${total:.2f})"
                )
    return notes

# ==============================================================================
# SCHEMA & AUDIT PROMPT WITH CHAIN-OF-THOUGHT (CoT)
# ==============================================================================
class AuditResult(BaseModel):
    thought_process: List[str] = Field(description="Step-by-step verification log resolving steps 1 through 5 sequentially.")
    violation: str = Field(description="Final compliance status. Must be exactly: 'None', 'Low', 'Medium', or 'High'.")
    reasoning: str = Field(description="Brief summary under 60 words for final reporting, referencing policy sections.")

AUDIT_PROMPT = ChatPromptTemplate.from_template(
    """You are an automated expense compliance auditor operating under policy HE-FIN-004 v3.1 merged.
Execute your evaluation sequentially following the structured verification steps below.

REFERENCE COMPLIANCE POLICY CONTEXT:
{policy_context}

CROSS-TRANSACTION CO-OCCURRENCES:
{cooccurrence_notes}

==================================================
👉 TARGET TRANSACTION TO AUDIT 👈
==================================================
Transaction ID: {id}
Employee Level: {level}
Department: {department}
Explicit Client Field: {client}
Merchant Venue: {merchant}
Claimed Category: {category}
Total Amount: ${amount}
Transaction Date: {date}
Justification Text: "{description}"

[PRE-CALCULATED MATHEMATICAL FACT BLOCK]:
- System Parsed Headcount: {python_headcount} person(s)
- System Calculated Per-Person Cost: ${python_per_person}
==================================================

MANDATORY CHAIN-OF-THOUGHT AUDITING STEPS:
You must execute and record observations for each step inside your `thought_process` list output:

STEP 1: ELIGIBILITY CHECK
- Locate the policy text section regulating the category "{category}" (e.g., Section 7.x).
- Verify if an employee at level "{level}" is permitted to claim this category at all.

STEP 2: CLIENT VALIDATION & VENUE ISOLATION
- Read the "Explicit Client Field". If it states a specific firm name, a client exists.
- Separate corporate clients from vendors. Remember that "{merchant}" is merely the point-of-sale venue, not an internal or external client.

STEP 3: THRESHOLD & PERMISSION RETRIEVAL
- Read the policy text and Appendix E to find the exact rule for "{category}" for a "{level}".
- Check if a "{level}" employee has the permission to submit this category at all. (e.g., If Juniors are banned from Entertainment, stop here—it is a High violation).
- Check if a "{level}" employee has internal hosting/party rights under Section 6 if there is "Internal / No Client".

STEP 4: DIRECT SEVERITY EVALUATION (TRUST THE PYTHON NUMBERS)
- You MUST use the pre-calculated `System Calculated Per-Person Cost` (${python_per_person}) for cap comparisons. Do NOT compare the total transaction Amount (${amount}) to a per-person cap.
- If the category has a per-person cap, evaluate: Is ${python_per_person} <= Cap? 
- If the employee lacks basic role permissions or hosting rights for this transaction type, it is an automatic violation, even if ${python_per_person} is below the cap.

STEP 5: STRUCTURAL ABUSE CHECK
- Evaluate the cross-transaction co-occurrence notes.

- If the transaction is technically under the math cap but has a completely vague justification (like "general purchase"), or is a clear personal expense hidden in a wrong category (like a gym membership in Entertainment), flag it as a violation based on policy non-compliance.
Output valid JSON matching this schema:
{{
  "thought_process": [
    "Step 1: [Observation here]",
    "Step 2: [Observation here]",
    "Step 3: [Observation here]",
    "Step 4: [Observation here]",
    "Step 5: [Observation here]"
  ],
  "violation": "None|Low|Medium|High",
  "reasoning": "Consolidated final reason under 60 words."
}}
"""
)

# ==============================================================================
# RUNTIME ENGINE LOOP
# ==============================================================================
def main():
    print("Loading primary compliance policy (PDF)...")
    policy_text = load_policy_text()

    print("Building structure-aware retriever...")
    retriever = StructureAwareRetriever()
    retriever.index_policy(policy_text)
    print(f"  Indexed {len(retriever.sections)} policy sections.")

    print("Loading source transaction batch...")
    with open(TRANSACTIONS_PATH, mode="r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"  Loaded {len(rows)} transactions.")

    print("Scanning data matrix for structural co-occurrences...")
    cooccurrences = find_cooccurrences(rows)

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0, format="json")
    parser = JsonOutputParser(pydantic_object=AuditResult)
    chain = AUDIT_PROMPT | llm | parser

    print(f"\n🚀 Executing sequential verification loop for {len(rows)} elements...")
    print(f"📁 Live streaming processed metrics to: {OUTPUT_PATH}")
    print("=" * 95)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "Transaction_ID", "Employee_Level", "Department", "Client", "Merchant",
            "Category", "Amount", "Date", "Justification",
            "violation", "reasoning", "execution_time_sec",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            start_time = time.perf_counter()
            tid = row["Transaction_ID"]
            cat = row["Category"]
            level = row["Employee_Level"]

            # --- DYNAMIC VENDOR SAFEGUARDS ---
            merchant_val = row.get("Merchant", "").strip()

            # --- DETACH MATHEMATICAL COMPUTATION (Python Execution) ---
            hc, per_person = parse_headcount_and_calc(row["Justification"], row["Amount"])

            always = retriever.forced_titles(ALWAYS_FORCED)
            category_specific = retriever.forced_titles(FORCED_SECTIONS_BY_CATEGORY.get(cat, []))
            semantic_query = f"{cat} claim by {level}: {row['Justification']}"
            semantic = retriever.semantic_context(semantic_query, k=2)

            titles = list(dict.fromkeys(always + category_specific + semantic))
            policy_context = retriever.render(titles)

            notes = cooccurrences.get(tid, [])
            notes_text = "\n".join(f"- {n}" for n in notes) if notes else "- None found."

            try:
                output = chain.invoke({
                    "id": tid,
                    "level": level,
                    "department": row["Department"],
                    "client": row["Client"],
                    "merchant": merchant_val,
                    "category": cat,
                    "amount": row["Amount"],
                    "date": row["Date"],
                    "description": row["Justification"],
                    "cooccurrence_notes": notes_text,
                    "policy_context": policy_context,
                    "python_headcount": hc,
                    "python_per_person": per_person
                })
                violation = output.get("violation", "None")
                reasoning = output.get("reasoning", "No reasoning returned.")
            except Exception as e:
                violation, reasoning = "Error", str(e)[:150]

            exec_time = round(time.perf_counter() - start_time, 2)
            writer.writerow({**row, "violation": violation, "reasoning": reasoning, "execution_time_sec": exec_time})
            csv_file.flush()
            print(f"⏳ [{idx}/{len(rows)}] Verified: {tid} | Violation: {violation:8s} | Took: {exec_time}s")

    print(f"\nExecution terminated. Output compiled at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()