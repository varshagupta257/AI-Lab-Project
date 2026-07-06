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

# ==========================================
# 1. LOAD POLICY PDF & OPTIMIZE RETRIEVAL
# ==========================================
print("📄 Loading policy document...")
pdf_loader = PyPDFLoader("/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf") 
policy_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
policy_chunks = text_splitter.split_documents(policy_docs)

print("🗄️ Indexing into local FAISS DB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(policy_chunks, embeddings)

# Keeping k at 5 to prevent overwhelming the context window
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 2. DEFINE STRUCTURAL SCHEMA
# ==========================================
class AuditOutput(BaseModel):
    RAG_Category: Literal["Meals", "Travel", "Software", "Office Supplies", "Entertainment", "Miscellaneous", "Wi-Fi/Connectivity"] = Field(
        description="The matching corporate expense category."
    )
    RAG_violation: Literal["High", "Medium", "Low", "None"] = Field(
        description="The matching risk compliance tier."
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

# ==========================================
# 3. INITIALIZE OLLAMA WITH LLAMA 8B (V5)
# ==========================================
# Switched model to llama3.1 and locked temperature to 0.0 for deterministic logic
llm = ChatOllama(
    model="llama3.1", 
    temperature=0.0,  
    format="json",
    timeout=45.0  
)
structured_engine = llm.with_structured_output(AuditOutput)

# ==========================================
# 4. SET UP REAL-TIME STREAMING CSV FILE
# ==========================================
csv_input_path = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2.csv"
output_filename = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output.csv"

# Safe string parsing to completely eliminate 'Jul 40' style date formatting issues
df = pd.read_csv(csv_input_path, sep=";", dtype=str)
total_rows = len(df)

headers = ["transaction ID", "Level", "Amount", "RAG Category", "RAG violation", "RAG Summary"]
with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print(f"\n🚀 Starting processing with Llama 8B [Strict Catch-All Guardrail]...")
print("=" * 70)

# ==========================================
# 5. ROW-BY-ROW STREAMING PROCESS
# ==========================================
for index, row in df.iterrows():
    row_start_time = time.time()
    current_row_num = index + 1
    
    raw_amount = str(row['Amount']).strip().replace(',', '.')
    submitted_category = str(row.get('Category', row.get('Expense_Category', 'Not Provided'))).strip()
    justification_str = str(row['Justification']).strip()
    merchant_str = str(row['Merchant']).strip()
    
    # Target lookup query
    lookup_query = f"{merchant_str} {justification_str} {submitted_category} expense limits restrictions"
    matched_chunks = retriever.invoke(lookup_query)
    context_str = "\n".join([doc.page_content for doc in matched_chunks])
    
    # Clean, concise prompt relying entirely on your new gold-standard auditing rule
    prompt_text = f"""You are a strict financial auditor validating employee expenses against the company policy document.

STRICT AUDIT RULE:
If an expense type, merchant type, or scenario is NOT explicitly mentioned or permitted anywhere in the provided policy context, you must treat it as an unapproved policy breach and mark the violation tier as High or Medium.

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
    
    rag_cat, rag_viol, rag_sum = "Miscellaneous", "High", "Failed parsing data correctly."
    
    try:
        result = structured_engine.invoke(prompt_text)
        rag_cat = result.RAG_Category
        rag_viol = result.RAG_violation
        rag_sum = result.RAG_Summary.strip()
                    
    except Exception as e:
        pass 

    # Save transaction block directly to disk
    with open(output_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row['Transaction_ID'], row['Employee_Level'], raw_amount, rag_cat, rag_viol, rag_sum])

    row_elapsed = time.time() - row_start_time
    print(f"⏳ [{current_row_num}/{total_rows}] Saved TXN: {row['Transaction_ID']} | Amount: {raw_amount} | Category: {rag_cat} | Violation: {rag_viol} | Took: {row_elapsed:.2f}s")

print("=" * 70)
print(f"🎉 Complete! Pure RAG processing finished using Llama 8B. Output compiled at '{output_filename}'")