import os
import json
import time
import csv
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

# =====================================================================
# 1. LOAD POLICY PDF & OPTIMIZE RETRIEVAL
# =====================================================================
print("📄 Loading policy document...")
pdf_loader = PyPDFLoader("/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf") 
policy_docs = pdf_loader.load()

# Split chunks tightly to keep semantic retrieval highly focused
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
policy_chunks = text_splitter.split_documents(policy_docs)

print("🗄️ Indexing into local FAISS DB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(policy_chunks, embeddings)

# Set retrieval to k=8 to maximize context coverage without spilling over limits
retriever = vector_store.as_retriever(search_kwargs={"k": 8})

# =====================================================================
# 2. STRUCTURAL SCHEMA WITH SEMANTIC FIELD HINTS
# =====================================================================
class AuditOutput(BaseModel):
    # Field descriptions act as inline semantic anchors for the LLM parser
    RAG_Category: Literal["Meals", "Travel", "Software", "Office Supplies", "Entertainment", "Miscellaneous", "Wi-Fi/Connectivity"] = Field(
        description="Meals=food/dining/restaurants/cafes; Travel=flights/hotels/taxis/rideshare; Software=subscriptions/tools/cloud services; Wi-Fi/Connectivity=internet/data plans."
    )
    RAG_violation: Literal["High", "Medium", "Low", "None"] = Field(
        description="The matching compliance risk tier dictated by the policy."
    )
    RAG_Summary: str = Field(
        description="Factual breakdown explaining compliance. Must be exactly 5 to 8 words maximum."
    )

    @field_validator('RAG_Summary')
    @classmethod
    def limit_summary_words(cls, v: str) -> str:
        words = v.split()
        if len(words) > 8:
            return " ".join(words[:7])
        return v

# =====================================================================
# 3. INITIALIZE OLLAMA WITH LLAMA 3.1 8B
# =====================================================================
# Switched to llama3.1 with locked temperature for strict deterministic auditing
llm = ChatOllama(
    model="llama3.1", 
    temperature=0.0,  
    format="json",
    timeout=45.0  
)
structured_engine = llm.with_structured_output(AuditOutput)

# =====================================================================
# 4. FILE PATH SETUPS & PRE-STREAMING PREPARATION
# =====================================================================
csv_input_path = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2.csv"
output_filename = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output.csv"

# Safe parsing explicitly string-typing rows to avoid timestamp corruption (e.g., 'Jul 40')
df = pd.read_csv(csv_input_path, sep=";", dtype=str)
total_rows = len(df)

headers = ["transaction ID", "Level", "Amount", "RAG Category", "RAG violation", "RAG Summary"]
with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print(f"\n🚀 Starting processing with Llama 3.1 (8B) [Strict Catch-All Guardrail active]...")
print("=" * 80)

# =====================================================================
# 5. ROW-BY-ROW STREAMING EVALUATION
# =====================================================================
for index, row in df.iterrows():
    row_start_time = time.time()
    current_row_num = index + 1
    
    raw_amount = str(row['Amount']).strip().replace(',', '.')
    submitted_category = str(row.get('Category', row.get('Expense_Category', 'Not Provided'))).strip()
    justification_str = str(row['Justification']).strip()
    merchant_str = str(row['Merchant']).strip()
    
    # Expanded query terms to extract matching budget limits into context
    lookup_query = f"{merchant_str} {justification_str} {submitted_category} meals food travel software compliance limits restrictions"
    matched_chunks = retriever.invoke(lookup_query)
    context_str = "\n".join([doc.page_content for doc in matched_chunks])
    
    prompt_text = f"""You are a strict financial auditor validating employee expenses against the company policy document.

STRICT AUDIT RULES:
1. Categorize accurately. Do NOT default to 'Miscellaneous' if the justification or merchant clearly references dining, food, restaurants, flights, hotels, or software.
2. Catch-All Guardrail: If an expense type, merchant, or scenario is NOT explicitly mentioned or permitted anywhere in the provided policy context, you must treat it as an unapproved breach and mark the violation tier as High or Medium.

POLICY CONTEXT:
{context_str}

TRANSACTION TO AUDIT:
- Transaction ID: {row['Transaction_ID']}
- Employee Level: {row['Employee_Level']}
- Stated Category: {submitted_category}
- Merchant: {merchant_str}
- Amount: {raw_amount}
- Justification: {justification_str}

Output your findings strictly following the schema layout, keeping the summary between 5 and 8 words.
"""
    
    # Default fallbacks in case of unexpected processing failure
    rag_cat, rag_viol, rag_sum = "Miscellaneous", "High", "Failed parsing data correctly."
    
    try:
        result = structured_engine.invoke(prompt_text)
        rag_cat = result.RAG_Category
        rag_viol = result.RAG_violation
        rag_sum = result.RAG_Summary.strip()
        
        # ─── PROGRAMMATIC CATCH-ALL GUARDRAIL ───────────────────────────
        # If LLM still chooses 'Miscellaneous', execute a runtime keyword filter override
        text_to_check = f"{justification_str.lower()} {merchant_str.lower()} {submitted_category.lower()}"
        if rag_cat == "Miscellaneous":
            if any(w in text_to_check for w in ["meal", "food", "dinner", "lunch", "restaurant", "cafe", "eat"]):
                rag_cat = "Meals"
            elif any(w in text_to_check for w in ["flight", "hotel", "taxi", "uber", "stay", "travel", "train"]):
                rag_cat = "Travel"
            elif any(w in text_to_check for w in ["software", "saas", "subscription", "licence", "aws", "cloud"]):
                rag_cat = "Software"
        # ────────────────────────────────────────────────────────────────
                    
    except Exception as e:
        # Gracefully handle validation/timeout exceptions without stalling execution
        pass 

    # Instantly stream row contents to storage
    with open(output_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row['Transaction_ID'], row['Employee_Level'], raw_amount, rag_cat, rag_viol, rag_sum])

    row_elapsed = time.time() - row_start_time
    print(f"⏳ [{current_row_num}/{total_rows}] Saved TXN: {row['Transaction_ID']} | Category: {rag_cat:<13} | Violation: {rag_viol:<6} | Took: {row_elapsed:.2f}s")

print("=" * 80)
print(f"🎉 Complete! Pure RAG pipeline finished execution on Llama 3.1. Output saved to '{output_filename}'")