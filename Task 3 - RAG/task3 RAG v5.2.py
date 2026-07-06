import os
import csv
import time
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field, field_validator

# LangChain & Ollama core components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

# =====================================================================
# 1. LOAD POLICY PDF & BUILD RAG VECTOR DATABASE
# =====================================================================
print("📄 Loading policy document...")
pdf_loader = PyPDFLoader("/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf") 
policy_docs = pdf_loader.load()

# Splitting documents into clean semantic chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
policy_chunks = text_splitter.split_documents(policy_docs)

print("🗄️ Indexing chunks into local FAISS vector store...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(policy_chunks, embeddings)

# Setting k=3 to keep context tightly bound to the specific transaction
retriever = vector_store.as_retriever(search_kwargs={"k": 9})

# =====================================================================
# 2. DEFINING PYDANTIC SCHEMA FOR STRUCTURED OUTPUT
# =====================================================================
class AuditOutput(BaseModel):
    RAG_Category: Literal["Meals", "Travel", "Software", "Office Supplies", "Entertainment", "Miscellaneous", "Wi-Fi/Connectivity"] = Field(
        description="The target category this transaction strictly belongs to based on context."
    )
    RAG_violation: Literal["High", "Medium", "Low", "None"] = Field(
        description="The compliance risk assignment dictated strictly by the policy boundaries."
    )
    RAG_Summary: str = Field(
        description="Factual summary detailing the audit logic. Must be exactly 5 to 8 words maximum."
    )

    @field_validator('RAG_Summary')
    @classmethod
    def limit_summary_words(cls, v: str) -> str:
        words = v.split()
        if len(words) > 8:
            return " ".join(words[:7])
        return v

# =====================================================================
# 3. INITIALIZE OLLAMA WITH LLAMA 3.1 (8B)
# =====================================================================
print("🤖 Initializing local Llama 3.1 structured engine...")
llm = ChatOllama(
    model="llama3.1", 
    temperature=0.0,  # Keeping it completely deterministic
    format="json",
    timeout=45.0  
)
structured_engine = llm.with_structured_output(AuditOutput)

# =====================================================================
# 4. LOAD TRANS DATA & PREPARE OUTPUT STORAGE
# =====================================================================
csv_input_path = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2.csv"
output_filename = "audited_transactions.csv"

# Reading dataframe natively
df = pd.read_csv(csv_input_path, sep=";", dtype=str)
total_rows = len(df)

# Initialize output CSV file with clean schema headers
with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Transaction_ID", "Level", "Amount", "RAG Category", "RAG violation", "RAG Summary"])

print(f"🚀 Starting pure RAG audit processing for {total_rows} transactions...")
print("=" * 80)

# =====================================================================
# 5. CORE RAG EVALUATION LOOP
# =====================================================================
for index, row in df.iterrows():
    row_start_time = time.time()
    current_row_num = index + 1
    
    txn_id = row['Transaction_ID']
    level = row['Employee_Level']
    amount = row['Amount']
    submitted_category = str(row.get('Category', row.get('Expense_Category', 'Not Provided'))).strip()
    justification = str(row['Justification']).strip()
    merchant = str(row['Merchant']).strip()
    
    # ─── THE RAG STEP ────────────────────────────────────────────────
    # Queries the local vector store using transaction properties to isolate target text
    lookup_query = f"{level} policy rules and spending limits for {submitted_category} {justification} {merchant}"
    matched_chunks = retriever.invoke(lookup_query)
    context_str = "\n".join([doc.page_content for doc in matched_chunks])
    # ─────────────────────────────────────────────────────────────────
    
    # Strictly grounding the model to the retrieved policy input string
    prompt_text = f"""You are a strict corporate financial auditor. Your task is to validate a single transaction against the provided company policy rules.

[EXTRACTED POLICY CONTEXT]
{context_str}

[TRANSACTION TO AUDIT]
- Transaction ID: {txn_id}
- Employee Level: {level}
- Submitted Category: {submitted_category}
- Merchant: {merchant}
- Amount: {amount}
- Justification: {justification}

[CRITICAL GROUNDING INSTRUCTIONS]
1. Map the transaction to its true category based on the Merchant and Justification (e.g., Meals, Software, Travel, Wi-Fi/Connectivity, or Miscellaneous).
2. Look ONLY at the policy rules matching that true category inside the [EXTRACTED POLICY CONTEXT]. Do NOT cross-mix categories or apply software limits to a meals expense.
3. STATED POLICY ONLY: Evaluate if the amount violates the allowed limit for this specific Employee Level using ONLY the text above. Do not use external knowledge or assume generic corporate limits. If the text does not specify a limit for this case, do not invent one.
4. If the transaction complies fully with the text provided, set violation to 'None'.
5. Output strictly using the JSON schema layout, keeping the summary under 8 words."""
    
    # Defaults if structured schema parsing experiences a data validation fault
    rag_cat = submitted_category if submitted_category != "Not Provided" else "Miscellaneous"
    rag_viol = "High"
    rag_sum = "Failed parsing structured response."
    
    try:
        result = structured_engine.invoke(prompt_text)
        rag_cat = result.RAG_Category
        rag_viol = result.RAG_violation
        rag_sum = result.RAG_Summary.strip()
    except Exception:
        pass 

    # Dynamic file-writing stream block to prevent runtime memory leakages
    with open(output_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([txn_id, level, amount, rag_cat, rag_viol, rag_sum])

    row_elapsed = time.time() - row_start_time
    print(f"⏳ [{current_row_num}/{total_rows}] Saved: {txn_id} | Category: {rag_cat:<15} | Violation: {rag_viol:<6} | Took: {row_elapsed:.2f}s")

print("=" * 80)
print(f"🏁 Complete! Pure RAG pipeline finished execution on Llama 3.1. Output saved to '{output_filename}'")