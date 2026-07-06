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
# Note: In future, use: from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# ==============================================================================
# v6.1.1 CONFIGURATION
# ==============================================================================
POLICY_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf"
TRANSACTIONS_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/expense_transactions-2.csv"
OUTPUT_PATH = "/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/rag_audit_output_v6.1.1_live.csv"

class AuditResult(BaseModel):
    category: str = Field(description="The detected expense category.")
    violation: str = Field(description="Violation level: 'None', 'Low', 'Medium', or 'High'.")
    reasoning: str = Field(description="Step-by-step logic citing the corporate policy context.")

# ==============================================================================
# v6.1.1 RETRIEVER
# ==============================================================================
class CustomParentChildRetriever:
    def __init__(self):
        print("Initializing embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.parent_store = {}
        
    def index_policies(self, policy_text: str):
        print("Slicing and indexing policy records...")
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        parent_chunks = parent_splitter.split_text(policy_text)
        child_docs = []
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            self.parent_store[parent_id] = parent_text
            for child in child_splitter.split_text(parent_text):
                child_docs.append(Document(page_content=child, metadata={"parent_id": parent_id}))
        self.vectorstore = FAISS.from_documents(child_docs, self.embeddings)
        
    def get_parent_context(self, query: str, k: int = 5) -> str:
        child_matches = self.vectorstore.similarity_search(query, k=k)
        unique_parent_ids = {doc.metadata.get("parent_id") for doc in child_matches if doc.metadata.get("parent_id")}
        return "\n\n---\n\n".join([self.parent_store[pid] for pid in unique_parent_ids])

# ==============================================================================
# v6.1.1 MAIN ENGINE (Sequential for Ollama Stability)
# ==============================================================================
if __name__ == "__main__":
    # Load PDF
    pdf_reader = PdfReader(POLICY_PATH)
    policy_content = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()])
    
    # Init Retriever
    retriever = CustomParentChildRetriever()
    retriever.index_policies(policy_content)
    
    # Load Transactions
    with open(TRANSACTIONS_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        real_transactions = [{"id": r.get('Transaction_ID'), "category": r.get('Category'), "amount": r.get('Amount'), "description": r.get('Justification')} for r in reader]

    # Init LLM & Parser
    llm = ChatOllama(model="llama3.1", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=AuditResult)
    prompt = ChatPromptTemplate.from_template(
        "You are a corporate auditor. Evaluate using this context: {context}\n"
        "Item: {description} (Amt: {amount}). \n"
        "1. Check policy adherence. 2. Flag (None/Low/Medium/High). 3. Reasoning.\n"
        "{format_instructions}"
    )
    chain = prompt | llm | parser

    print(f"\n🚀 [v6.1.1] Starting sequential audit on {len(real_transactions)} items...")
    
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['id', 'category', 'amount', 'description', 'violation', 'reasoning', 'execution_time_sec'])
        writer.writeheader()

        for idx, txn in enumerate(real_transactions, 1):
            start_time = time.perf_counter()
            search_query = f"Policy rules for {txn['category']}: {txn['description']}"
            context = retriever.get_parent_context(search_query, k=5)
            
            try:
                output = chain.invoke({
                    "context": context, "description": txn['description'], "amount": txn['amount'], 
                    "format_instructions": parser.get_format_instructions()
                })
                violation = output.get("violation", "None")
                reasoning = output.get("reasoning", "No reasoning.")
            except Exception as e:
                violation, reasoning = "Error", str(e)[:50]
                
            exec_time = round(time.perf_counter() - start_time, 2)
            writer.writerow({**txn, 'violation': violation, 'reasoning': reasoning, 'execution_time_sec': exec_time})
            csv_file.flush()
            print(f"⏳ [{idx}/{len(real_transactions)}] Saved: {txn['id']} | Violation: {violation} | Time: {exec_time}s")