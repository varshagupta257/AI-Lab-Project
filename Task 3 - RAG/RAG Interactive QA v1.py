from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ==========================================
# 1. LOAD POLICY PDF & BUILD LOCAL FAISS DB (V4 CONFIG)
# ==========================================
print("📄 Loading policy document...")
pdf_loader = PyPDFLoader("/Users/varshagupta/Desktop/VS Code/AI Lab/AI Lab Project/Task 3 - RAG/data/Halcyon_Expense_Policy_v3.1.pdf") 
policy_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
policy_chunks = text_splitter.split_documents(policy_docs)

print("🗄️ Indexing into local FAISS DB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(policy_chunks, embeddings)

# Set retrieval to k=5 exactly as configured in V4
retriever = vector_store.as_retriever(search_kwargs={"k": 7})

# ==========================================
# 2. INITIALIZE MISTRAL (V4 CONFIG)
# ==========================================
llm = ChatOllama(model="mistral", temperature=0.1)

print("\n🚀 RAG Auditor Interface Initialized Successfully.")
print("💡 Type your compliance question below. Type 'exit' or 'quit' to close the program.\n")
print("=" * 70)

# ==========================================
# 3. INTERACTIVE AUDITOR LOOP
# ==========================================
while True:
    # Capture user input dynamically
    question = input("\n🔍 Enter compliance question to audit: ").strip()
    
    # Check for exit commands
    if question.lower() in ['exit', 'quit', 'q']:
        print("👋 Closing Auditor Interface. Goodbye!")
        break
        
    if not question:
        continue

    print("🤖 Running RAG Audit Analysis...")
    
    try:
        # Retrieve context from FAISS based on dynamic query
        matched_chunks = retriever.invoke(question)
        context_str = "\n\n".join([doc.page_content for doc in matched_chunks])

        # Strict formatting instructions matching your grading expectations
        qa_prompt = f"""You are a strict corporate compliance officer. 
Analyze the transaction against the provided policy context below.

CRITICAL OUTPUT INSTRUCTIONS:
1. Your response must start exactly with one of these three verdicts: [Approved], [Declined], or [Needs Human Review].
2. Following the verdict, provide a comprehensive compliance explanation that details any role-specific caps, per-person math limits, or structural violations found in the context.
3. Your entire explanation text MUST be approximately 50 words in length. Do not write a short sentence.

POLICY CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:"""

        # Generate response from Mistral
        response = llm.invoke(qa_prompt)
        print("\n" + "=" * 40 + " AUDIT REPORT " + "=" * 40)
        print(response.content)
        print("=" * 94)
        
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")