import os
import json
import time
import csv
import pandas as pd
from typing import Literal  # Enforces a rigid set of options in the schema
from pydantic import BaseModel, Field, field_validator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ==========================================
# 1. LOAD POLICY PDF & BUILD LOCAL VECTOR DB
# ==========================================
print("📄 Loading policy document using PyPDFLoader...")
pdf_loader = PyPDFLoader("/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf") 
policy_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
policy_chunks = text_splitter.split_documents(policy_docs)

print("🗄️ Embedding policy sections locally into Chroma...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(policy_chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 2. DEFINE STRUCTURAL SCHEMA & STRICT VALIDATION
# ==========================================
class AuditOutput(BaseModel):
    # Literal forces the LLM's JSON engine to pick ONLY one of these exact short strings
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
            return " ".join(words[:7])  # Clean fallback truncation
        return v

# Initialize the model with programmatic JSON structural enforcement
llm = ChatOllama(
    model="llama3.2:3b", 
    temperature=0.0, # Zeroed completely out to destroy loose conversational habits
    format="json",
    timeout=30.0  
)
structured_engine = llm.with_structured_output(AuditOutput)

# ==========================================
# 3. SET UP REAL-TIME STREAMING CSV FILE
# ==========================================
csv_input_path = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2.csv"
output_filename = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output.csv"

df = pd.read_csv(csv_input_path, sep=";")
total_rows = len(df)

# Initialize output file with headers immediately
headers = ["transaction ID", "Level", "RAG Category", "RAG violation", "RAG Summary"]
with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print(f"\n🚀 Starting streaming batch processing for {total_rows} transactions...")
print(f"📁 Clean real-time entries saving directly to: {output_filename}")
print("=" * 70)

# ==========================================
# 4. ROW-BY-ROW STREAMING PROCESS
# ==========================================
for index, row in df.iterrows():
    row_start_time = time.time()
    current_row_num = index + 1
    
    # Context-aware lookup query
    lookup_query = f"{row['Merchant']} {row['Justification']} expense rules"
    matched_chunks = retriever.invoke(lookup_query)
    context_str = "\n".join([doc.page_content for doc in matched_chunks])
    
    # Restructured prompt framework to remove text leakage
    data_payload = {
        "system_instruction": (
            "You are a strict financial auditor. Match the transaction data to the provided corporate policy context. "
            "Select an exact category value, decide on the strict risk assessment tier, and write an absolute factual "
            "summary of exactly 5 to 8 words. Do not quote irrelevant rules or add side remarks."
        ),
        "policy_context": context_str,
        "transaction_to_audit": {
            "ID": row['Transaction_ID'],
            "Merchant": row['Merchant'],
            "Amount": row['Amount'],
            "Justification": row['Justification'],
            "Employee_Level": row['Employee_Level']
        }
    }
    
    # Default fallbacks
    rag_cat, rag_viol, rag_sum = "Miscellaneous", "High", "Failed parsing internal data correctly."
    
    try:
        result = structured_engine.invoke(json.dumps(data_payload))
        rag_cat = result.RAG_Category
        rag_viol = result.RAG_violation
        rag_sum = result.RAG_Summary.strip()
    except Exception as e:
        pass 

    # Append directly to file on disk immediately
    with open(output_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row['Transaction_ID'], row['Employee_Level'], rag_cat, rag_viol, rag_sum])

    row_elapsed = time.time() - row_start_time
    print(f"⏳ [{current_row_num}/{total_rows}] Saved TXN: {row['Transaction_ID']} | Category: {rag_cat} | Took: {row_elapsed:.2f}s")

print("=" * 70)
print(f"🎉 Complete! Clean data output compiled into '{output_filename}'")