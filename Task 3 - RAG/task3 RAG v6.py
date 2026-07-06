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
TRANSACTIONS_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2.csv"
OUTPUT_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output_v6_live.csv"

# ==============================================================================
# 1. DEFINE STRUCTURED AUDIT OUTPUT SCHEMA
# ==============================================================================
class AuditResult(BaseModel):
    category: str = Field(description="The detected expense category.")
    violation: str = Field(description="Violation level: 'None', 'Low', 'Medium', or 'High'.")
    reasoning: str = Field(description="Step-by-step logic citing the corporate policy context.")

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
        
    def get_parent_context(self, query: str, k: int = 5) -> str:
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
# 3. CONSTRUCT THE RUNTIME PIPELINE WITH REAL-TIME CSV STREAMING
# ==============================================================================
def run_v6_audit_system(transactions: List[Dict[str, Any]], policy_text: str, llm_model: Any, output_csv_path: str):
    
    retriever = CustomParentChildRetriever()
    retriever.index_policies(policy_text)

    parser = JsonOutputParser(pydantic_object=AuditResult)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert corporate financial auditor auditing employee expense reports.\n"
        "Evaluate the transaction below using ONLY the provided policy context.\n\n"
        "--- CORPORATE POLICY CONTEXT ---\n"
        "{context}\n\n"
        "--- TRANSACTION TO EVALUATE ---\n"
        "ID: {id}\n"
        "Category: {category}\n"
        "Amount: {amount}\n"
        "Description: {description}\n\n"
        "--- SYSTEM INSTRUCTIONS ---\n"
        "1. Evaluate if this item violates the specific policy conditions.\n"
        "2. If the context mentions policy boundaries for this item type, apply them strictly.\n"
        "3. Only output 'None' if the item completely satisfies the retrieved rules, or if "
        "the category is completely missing from the context.\n"
        "{format_instructions}"
    )

    chain = prompt | llm_model | parser

    print(f"\n🚀 Starting V6 Custom Parent-Child [500/200] loop for {len(transactions)} items...")
    print(f"📁 Live streaming results to: {output_csv_path}")
    print("=" * 90)

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['id', 'category', 'amount', 'description', 'violation', 'reasoning', 'execution_time_sec']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, txn in enumerate(transactions, 1):
            start_time = time.perf_counter()
            
            search_query = f"Expense policy rules for category {txn['category']}: {txn['description']}"
            context_string = retriever.get_parent_context(search_query, k=5)
            
            try:
                output = chain.invoke({
                    "context": context_string,
                    "id": txn["id"],
                    "category": txn["category"],
                    "amount": txn["amount"],
                    "description": txn["description"],
                    "format_instructions": parser.get_format_instructions()
                })
                violation = output.get("violation", "None")
                reasoning = output.get("reasoning", "No reasoning provided.")
            except Exception as e:
                violation = "Error"
                reasoning = f"Parsing Failed: {str(e)[:100]}"
                
            execution_time = round(time.perf_counter() - start_time, 2)
            
            writer.writerow({
                'id': txn['id'],
                'category': txn['category'],
                'amount': txn['amount'],
                'description': txn['description'],
                'violation': violation,
                'reasoning': reasoning,
                'execution_time_sec': execution_time
            })
            
            csv_file.flush()
            
            print(f"⏳ [{idx}/{len(transactions)}] Saved: {txn['id']} | "
                  f"Category: {txn['category']:<18} | "
                  f"Violation: {violation:<8} | "
                  f"Took: {execution_time}s")

# ==============================================================================
# 4. DATA LOADER BRIDGE & RUNTIME EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1. Parse policy text from PDF
    if not os.path.exists(POLICY_PATH):
        raise FileNotFoundError(f"Could not find the policy PDF file at: {POLICY_PATH}")
    
    print(f"Opening policy document: {os.path.basename(POLICY_PATH)}")
    pdf_reader = PdfReader(POLICY_PATH)
    policy_content = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            policy_content += text + "\n"
            
    # 2. Parse real records with smart auto-detection
    if not os.path.exists(TRANSACTIONS_PATH):
        raise FileNotFoundError(f"Could not find the transactions CSV file at: {TRANSACTIONS_PATH}")
        
    print(f"Loading transaction dataset: {os.path.basename(TRANSACTIONS_PATH)}")
    real_transactions = []
    
    # Step A: Sniff the correct delimiter
    with open(TRANSACTIONS_PATH, mode='r', encoding='utf-8-sig') as test_f:
        first_line = test_f.readline()
        detected_delimiter = ';' if ';' in first_line and first_line.count(';') > first_line.count(',') else ','
    
    with open(TRANSACTIONS_PATH, mode='r', encoding='utf-8-sig') as f:
        csv_reader = csv.DictReader(f, delimiter=detected_delimiter)
        raw_headers = [h.strip() for h in csv_reader.fieldnames if h]
        
        if not raw_headers:
            raise ValueError("CSV structure is completely unreadable or empty.")
            
        # Step B: Smart multi-layer mapping rules for financial/accounting nomenclature
        # Prioritize descriptions first to isolate descriptive phrases from categories
        desc_header = next((h for h in raw_headers if any(x in h.lower() for x in ['desc', 'det', 'memo', 'purpose', 'text', 'item', 'transaction'])), None)
        
        # Look for structural category fields (excluding the narrative description column)
        cat_header = next((h for h in raw_headers if any(x in h.lower() for x in ['cat', 'type', 'class', 'group', 'account', 'head', 'gl']) and h != desc_header), None)
        
        amt_header = next((h for h in raw_headers if any(x in h.lower() for x in ['am', 'val', 'cost', 'price', 'eur', 'usd', 'total', 'spent'])), None)
        id_header = next((h for h in raw_headers if any(x in h.lower() for x in ['id', 'txn', 'num', 'code'])), None)
        
        # Display the column mapping diagnostic table for total verification
        print("\n🔍 CSV COLUMN MAPPING DIAGNOSTIC:")
        print(f"  • Raw CSV Headers Detected: {raw_headers}")
        print(f"  • Assigned 'id'          -> {id_header}")
        print(f"  • Assigned 'category'    -> {cat_header}")
        print(f"  • Assigned 'amount'      -> {amt_header}")
        print(f"  • Assigned 'description' -> {desc_header}")
        print("=" * 90)
        
        # Step C: Parse rows with strict key lookups and layout fallbacks
        for row in csv_reader:
            row_values = [row.get(h, "").strip() for h in csv_reader.fieldnames if h]
            if not any(row_values):
                continue
                
            txn_id = row.get(id_header).strip() if id_header else (row_values[0] if len(row_values) > 0 else "UNKNOWN")
            category = row.get(cat_header).strip() if cat_header else (row_values[1] if len(row_values) > 1 else "Misc")
            amount = row.get(amt_header).strip() if amt_header else (row_values[2] if len(row_values) > 2 else "0.00")
            description = row.get(desc_header).strip() if desc_header else (row_values[3] if len(row_values) > 3 else "")
            
            # Formatting Guardrail: If category and description targeted the exact same large field,
            # extract a clean prefix for the category so the vector search query is precise.
            if category == description and len(description) > 30:
                category = " ".join(description.split()[:3]) + "..."
            
            real_transactions.append({
                "id": txn_id,
                "category": category,
                "amount": amount,
                "description": description
            })

    # 3. Fire up the local LLM and start processing
    print("Connecting to local Llama 3.1 model...")
    local_llama = ChatOllama(model="llama3.1", temperature=0.0)
    
    run_v6_audit_system(real_transactions, policy_content, local_llama, OUTPUT_PATH)