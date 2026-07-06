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
# 1. LOAD POLICY PDF & BUILD LOCAL FAISS DB
# ==========================================
print("📄 Loading policy document using PyPDFLoader...")
pdf_loader = PyPDFLoader("/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf") 
policy_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=350)
policy_chunks = text_splitter.split_documents(policy_docs)

print("🗄️ Embedding policy sections locally into FAISS...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(policy_chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 9})

# ==========================================
# 2. DEFINE STRUCTURAL SCHEMA & STRICT VALIDATION
# ==========================================
class AuditOutput(BaseModel):
    RAG_Category: Literal["Meals", "Travel", "Software", "Office Supplies", "Entertainment", "Miscellaneous", "Wi-Fi/Connectivity"] = Field(
        description="The exact matching corporate expense category from the policy options."
    )
    RAG_violation: Literal["High", "Medium", "Low", "None"] = Field(
        description="The matching risk compliance tier based on the rules."
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
# 3. INITIALIZE OLLAMA WITH MISTRAL & TEMP 0.7
# ==========================================
llm = ChatOllama(
    model="mistral", 
    temperature=0.1, 
    format="json",
    timeout=30.0  
)
structured_engine = llm.with_structured_output(AuditOutput)

# ==========================================
# 4. SET UP REAL-TIME STREAMING CSV FILE
# ==========================================
csv_input_path = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2.csv"
output_filename = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output.csv"

df = pd.read_csv(csv_input_path, sep=";")
total_rows = len(df)

# Added "Amount" to headers
headers = ["transaction ID", "Level", "Amount", "RAG Category", "RAG violation", "RAG Summary"]
with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print(f"\n🚀 Starting streaming batch processing using Mistral via FAISS (k=9)...")
print(f"📁 Clean real-time entries saving directly to: {output_filename}")
print("=" * 70)

# ==========================================
# 5. ROW-BY-ROW STREAMING PROCESS
# ==========================================
for index, row in df.iterrows():
    row_start_time = time.time()
    current_row_num = index + 1
    
    # Extract employee's stated category (handling potential column naming edge cases)
    submitted_category = row.get('Category', row.get('Expense_Category', 'Not Provided'))
    
    # Context-aware lookup query using the merchant, justification, and original category
    lookup_query = f"{row['Merchant']} {row['Justification']} {submitted_category} expense rules"
    matched_chunks = retriever.invoke(lookup_query)
    context_str = "\n".join([doc.page_content for doc in matched_chunks])
    
    # Prompt explicitly provides the employee's submitted category to compare against policy rules
    prompt_text = f"""You are a strict financial auditor. Match the transaction data to the provided corporate policy context. 
Analyze the rules and exceptions dynamically from the context below to find discrepancies between employee levels, corporate categories, caps, or prohibited practices.

POLICY CONTEXT:
{context_str}

TRANSACTION TO AUDIT:
- Transaction ID: {row['Transaction_ID']}
- Employee Level: {row['Employee_Level']}
- Stated/Submitted Category: {submitted_category}
- Merchant: {row['Merchant']}
- Amount: {row['Amount']}
- Justification: {row['Justification']}

Output your findings strictly following the schema layout, keeping the summary between 5 and 8 words.
"""
    
    rag_cat, rag_viol, rag_sum = "Miscellaneous", "High", "Failed parsing internal data correctly."
    
    try:
        result = structured_engine.invoke(prompt_text)
        rag_cat = result.RAG_Category
        rag_viol = result.RAG_violation
        rag_sum = result.RAG_Summary.strip()
    except Exception as e:
        pass 

    # Saved with the Amount column tracking directly on disk
    with open(output_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row['Transaction_ID'], row['Employee_Level'], row['Amount'], rag_cat, rag_viol, rag_sum])

    row_elapsed = time.time() - row_start_time
    print(f"⏳ [{current_row_num}/{total_rows}] Saved TXN: {row['Transaction_ID']} | Amount: {row['Amount']} | Category: {rag_cat} | Violation: {rag_viol} | Took: {row_elapsed:.2f}s")

print("=" * 70)
print(f"🎉 Complete! Clean data output compiled into '{output_filename}'")