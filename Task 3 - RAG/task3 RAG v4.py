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
# 1. LOAD POLICY PDF & RESTORE V1.4 CHUNK LIMITS
# ==========================================
print("📄 Loading policy document using PyPDFLoader...")
pdf_loader = PyPDFLoader("/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf") 
policy_docs = pdf_loader.load()

# Restored chunk size and overlap back to the sweet spot from V1.4
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
policy_chunks = text_splitter.split_documents(policy_docs)

print("🗄️ Embedding policy sections locally into FAISS...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(policy_chunks, embeddings)

# Adjusted k back to 7 to prevent overwhelming Mistral with redundant context text
retriever = vector_store.as_retriever(search_kwargs={"k": 7})

# ==========================================
# 2. DEFINE STRUCTURAL SCHEMA WITH STRICT CATEGORY ANTI-BIAS
# ==========================================
class AuditOutput(BaseModel):
    RAG_Category: Literal["Meals", "Travel", "Software", "Office Supplies", "Entertainment", "Miscellaneous", "Wi-Fi/Connectivity"] = Field(
        description=(
            "The matching corporate expense category. CRITICAL: You must explicitly evaluate if the transaction "
            "maps to Meals, Travel, Software, Office Supplies, Entertainment, or Wi-Fi/Connectivity first. "
            "Only choose 'Miscellaneous' as an absolute last resort if it has absolutely zero conceptual overlap with the others."
        )
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
# 3. INITIALIZE OLLAMA WITH MISTRAL (STRICT TEMP)
# ==========================================
llm = ChatOllama(
    model="mistral", 
    temperature=0.1,  # Kept at 0.1 to eliminate random hallucinations seen in V3
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

headers = ["transaction ID", "Level", "Amount", "RAG Category", "RAG violation", "RAG Summary"]
with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

print(f"\n🚀 Starting streaming processing for V4 [Anti-Miscellaneous Guardrails Built-in]...")
print(f"📁 Dashboard ready entries saving directly to: {output_filename}")
print("=" * 70)

# ==========================================
# 5. ROW-BY-ROW STREAMING PROCESS
# ==========================================
for index, row in df.iterrows():
    row_start_time = time.time()
    current_row_num = index + 1
    
    submitted_category = row.get('Category', row.get('Expense_Category', 'Not Provided'))
    
    # Target lookup query to ensure the embedding engine pulls high-relevance rules
    lookup_query = f"{row['Merchant']} {row['Justification']} {submitted_category} corporate expense policy limits restrictions"
    matched_chunks = retriever.invoke(lookup_query)
    context_str = "\n".join([doc.page_content for doc in matched_chunks])
    
    # Prompt fully updated with explicit hierarchical rules to completely disable lazy grouping bias
    prompt_text = f"""You are a strict financial auditor checking transaction compliance against corporate rules. 

CRITICAL CATEGORY RULES:
1. Examine the 'Stated/Submitted Category' and the 'Merchant/Justification' fields below.
2. If the transaction involves food, restaurants, or team dinners, you MUST map it to 'Meals'.
3. If it involves flights, hotels, taxi/uber, or transit, you MUST map it to 'Travel'.
4. If it involves licenses, cloud tools, SaaS, or digital subscriptions, you MUST map it to 'Software'.
5. If it involves paper, keyboards, desks, or stationery, you MUST map it to 'Office Supplies'.
6. If it involves team events, client outings, or tickets, you MUST map it to 'Entertainment'.
7. If it involves internet, phone bills, or router setups, you MUST map it to 'Wi-Fi/Connectivity'.
8. Only output 'Miscellaneous' if it completely evades all definitions above. Do not default to Miscellaneous for safety.

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
    
    # Default fallback values if parsing fails completely
    rag_cat, rag_viol, rag_sum = "Miscellaneous", "High", "Failed parsing internal data correctly."
    
    try:
        result = structured_engine.invoke(prompt_text)
        rag_cat = result.RAG_Category
        rag_viol = result.RAG_violation
        rag_sum = result.RAG_Summary.strip()
        
        # --- PROGRAMMATIC GUARDRAIL BLOCK ---
        # If Mistral still stubbornly chooses "Miscellaneous" but the original submitted category
        # was perfectly valid, we force-override it back to prevent your dashboard from breaking.
        VALID_TARGETS = ["Meals", "Travel", "Software", "Office Supplies", "Entertainment", "Wi-Fi/Connectivity"]
        if rag_cat == "Miscellaneous" and submitted_category in VALID_TARGETS:
            rag_cat = submitted_category
            
    except Exception as e:
        pass 

    # Save tracking information on disk
    with open(output_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row['Transaction_ID'], row['Employee_Level'], row['Amount'], rag_cat, rag_viol, rag_sum])

    row_elapsed = time.time() - row_start_time
    print(f"⏳ [{current_row_num}/{total_rows}] Saved TXN: {row['Transaction_ID']} | Amount: {row['Amount']} | Category: {rag_cat} | Violation: {rag_viol} | Took: {row_elapsed:.2f}s")

print("=" * 70)
print(f"🎉 Complete! Clean V4 dataset compiled and balanced for your dashboard in '{output_filename}'")