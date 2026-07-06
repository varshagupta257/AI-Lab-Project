import os
import time
import uuid
import csv
import warnings
from typing import List, Dict, Any

# Mute LangChain's deprecation warnings
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
# LOCAL FILE PATH CONFIGURATION
# ==============================================================================
POLICY_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf"
TRANSACTIONS_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2withCAT.csv"
OUTPUT_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output_v6_live.csv"

# ==============================================================================
# 1. DEFINE STRUCTURED AUDIT OUTPUT SCHEMA (With Category Re-evaluation)
# ==============================================================================
class AuditResult(BaseModel):
    corrected_category: str = Field(description="The standard, validated corporate policy category determined by analyzing the description (e.g., 'Meals & Entertainment', 'Wellness', 'Software & Subscriptions').")
    violation: str = Field(description="Violation level: 'None', 'Low', 'Medium', or 'High'.")
    reasoning: str = Field(description="Explanation of why this category was assigned and how the transaction details map to specific policy restrictions or allowances.")

# ==============================================================================
# 2. CUSTOM PARENT-CHILD RETRIEVER
# ==============================================================================
class CustomParentChildRetriever:
    def __init__(self):
        print("Initializing embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.parent_store = {}
        
    def index_policies(self, policy_text: str):
        print("Slicing and indexing policy records natively...")
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        
        parent_chunks = parent_splitter.split_text(policy_text)
        child_docs = []
        
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            self.parent_store[parent_id] = parent_text
            
            children = child_splitter.split_text(parent_text)
            for child in children:
                child_docs.append(Document(page_content=child, metadata={"parent_id": parent_id}))
                
        self.vectorstore = FAISS.from_documents(child_docs, self.embeddings)
        
    def get_parent_context(self, query: str, k: int = 6) -> str:
        child_matches = self.vectorstore.similarity_search(query, k=k)
        
        unique_parent_ids = set()
        parent_texts = []
        
        for doc in child_matches:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in unique_parent_ids:
                unique_parent_ids.add(pid)
                parent_texts.append(self.parent_store[pid])
                
        return "\n\n---\n\n".join(parent_texts)

# ==============================================================================
# 3. RUNTIME PIPELINE WITH REAL-TIME STREAMING
# ==============================================================================
def run_v6_audit_system(transactions: List[Dict[str, Any]], policy_text: str, llm_model: Any, output_csv_path: str):
    
    retriever = CustomParentChildRetriever()
    retriever.index_policies(policy_text)

    parser = JsonOutputParser(pydantic_object=AuditResult)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert corporate forensic auditor. Employees often intentionally or accidentally miscategorize their expenses.\n"
        "Your first task is to re-evaluate and correct the category based strictly on the transaction description and the policy definitions.\n\n"
        "--- RETRIEVED CORPORATE POLICY CONTEXT ---\n"
        "{context}\n\n"
        "--- EMPLOYEE TRANSACTION DATA ---\n"
        "ID: {id}\n"
        "Employee Claimed Category: {employee_category}\n"
        "Amount: {amount}\n"
        "Item Description/Justification: {description}\n\n"
        "--- CORE AUDIT DIRECTIVES ---\n"
        "1. Identify the TRUE category from the policy text that matches the 'Item Description/Justification'. Ignore the 'Employee Claimed Category' if it is inaccurate.\n"
        "2. Once the true category is established, audit the 'Amount' and 'Description' against that specific category's spending caps, limits, or approval mandates.\n"
        "3. If an item belongs to an entirely non-reimbursable category (like personal wellness/gyms under strict business rules, or general personal items), flag it as a 'High' violation.\n"
        "{format_instructions}"
    )

    chain = prompt | llm_model | parser

    print(f"\n🚀 Starting V6 Category Verification & Audit loop for {len(transactions)} items...")
    print(f"📁 Live streaming corrected matrix to: {output_csv_path}")
    print("=" * 115)

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['id', 'employee_category', 'corrected_category', 'amount', 'description', 'violation', 'reasoning', 'execution_time_sec']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, txn in enumerate(transactions, 1):
            start_time = time.perf_counter()
            
            # Hybrid search query ensures the correct policy chunks are fetched even if the employee category is wrong
            search_query = f"Expense guidelines parameters for {txn['employee_category']} or item described as: {txn['description']}"
            context_string = retriever.get_parent_context(search_query, k=6)
            
            try:
                output = chain.invoke({
                    "context": context_string,
                    "id": txn["id"],
                    "employee_category": txn["employee_category"],
                    "amount": txn["amount"],
                    "description": txn["description"],
                    "format_instructions": parser.get_format_instructions()
                })
                corrected_cat = output.get("corrected_category", "Unclassified")
                violation = output.get("violation", "None")
                reasoning = output.get("reasoning", "No reasoning provided.")
            except Exception as e:
                corrected_cat = "Parsing Error"
                violation = "Error"
                reasoning = f"Llama Output Parsing Failed: {str(e)[:100]}"
                
            execution_time = round(time.perf_counter() - start_time, 2)
            
            writer.writerow({
                'id': txn['id'],
                'employee_category': txn['employee_category'],
                'corrected_category': corrected_cat,
                'amount': txn['amount'],
                'description': txn['description'],
                'violation': violation,
                'reasoning': reasoning,
                'execution_time_sec': execution_time
            })
            
            csv_file.flush()
            
            # Clear visual check showing original vs corrected categorization
            print(f"⏳ [{idx}/{len(transactions)}] Saved: {txn['id']} | "
                  f"Claimed: {txn['employee_category'][:20]:<20} -> "
                  f"True Class: {corrected_cat[:22]:<22} | "
                  f"Violation: {violation:<6} | "
                  f"Took: {execution_time}s")

# ==============================================================================
# 4. DATA PARSER BRIDGE
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(POLICY_PATH) or not os.path.exists(TRANSACTIONS_PATH):
        raise FileNotFoundError("Verify your local file path definitions match your setup.")
    
    print(f"Extracting policy text from master PDF...")
    pdf_reader = PdfReader(POLICY_PATH)
    policy_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            
    print(f"Opening reference file: {os.path.basename(TRANSACTIONS_PATH)}")
    real_transactions = []
    
    with open(TRANSACTIONS_PATH, mode='r', encoding='utf-8-sig') as f:
        first_line = f.readline()
        delim = ';' if ';' in first_line and first_line.count(';') > first_line.count(',') else ','
        f.seek(0)
        
        csv_reader = csv.DictReader(f, delimiter=delim)
        raw_headers = [h.strip() for h in csv_reader.fieldnames if h]
        
        id_header = next((h for h in csv_reader.fieldnames if any(x in h.lower() for x in ['id', 'txn', 'num'])), None)
        amt_header = next((h for h in csv_reader.fieldnames if any(x in h.lower() for x in ['am', 'val', 'cost', 'price', 'total'])), None)
        desc_header = next((h for h in csv_reader.fieldnames if any(x in h.lower() for x in ['desc', 'det', 'memo', 'purpose', 'text', 'justification'])), None)
        cat_header = next((h for h in csv_reader.fieldnames if any(x in h.lower() for x in ['cat', 'type', 'class', 'group', 'account', 'head']) and h != desc_header), None)
        
        # Absolute structural positional fallbacks
        if not id_header: id_header = raw_headers[0]
        if not cat_header: cat_header = raw_headers[1]
        if not amt_header: amt_header = raw_headers[2]
        if not desc_header: desc_header = raw_headers[3] if len(raw_headers) > 3 else raw_headers[1]

        print("\n🔍 STRICT COLUMN MAPPING VERIFIED:")
        print(f"  • Transaction ID Column    -> '{id_header}'")
        print(f"  • Employee Category Column -> '{cat_header}'")
        print(f"  • Transaction Amount Column -> '{amt_header}'")
        print(f"  • Item Description Column   -> '{desc_header}'")
        print("=" * 115)
        
        for row in csv_reader:
            vals = [row.get(h, "").strip() for h in csv_reader.fieldnames if h]
            if not any(vals):
                continue
                
            real_transactions.append({
                "id": str(row.get(id_header, "")).strip(),
                "employee_category": str(row.get(cat_header, "General")).strip(),
                "amount": str(row.get(amt_header, "0.00")).strip(),
                "description": str(row.get(desc_header, "")).strip()
            })

    print("Connecting to local Llama 3.1 engine...")
    local_llama = ChatOllama(model="llama3.1", temperature=0.0)
    
    run_v6_audit_system(real_transactions, policy_content, local_llama, OUTPUT_PATH)
