import re
from ollama import chat
from config import OLLAMA_MODEL
from chatbot.pdf_reader import get_document_text
from chatbot.vectorstore import get_or_create_retriever, query_structure_aware_rag

SYSTEM_PROMPT = """
You are the AuditAI Expense Compliance Auditor.

ROLE & INSTRUCTIONS:
You evaluate employee expense submissions against corporate compliance policy HE-FIN-004. 
Analyze the provided transaction details step-by-step and deliver a clear, professional assessment.

REFERENCE POLICY CONTEXT:
{policy_context}

--------------------------------------------------
MANDATORY AUDITING STEPS:
Evaluate the expense using the following five checks:

STEP 1: ELIGIBILITY CHECK
- Verify if an employee at this level is permitted to claim this specific category under the policy rules.

STEP 2: CLIENT VALIDATION & VENUE ISOLATION
- Identify if an external corporate client was present. Note that the merchant venue is just where the money was spent, not the client.

STEP 3: THRESHOLD & PERMISSION RETRIEVAL
- Check the exact spending limits or caps allowed for this employee level in the policy (including specific meal or entertainment caps).

STEP 4: SPENDING COMPLIANCE EVALUATION
- Look at the "Calculated Cost Per Person" provided in the transaction data.
- Do NOT perform division yourself. Use the exact "Calculated Cost Per Person" value provided.
- Compare this "Calculated Cost Per Person" against the policy cap found in Step 3.
- CRITICAL MATH RULE: Ensure you perform the mathematical inequality check correctly. For example, $48.90 is LESS than $150.00. If the cost per person is lower than or equal to the cap, it does NOT exceed the limit!

STEP 5: JUSTIFICATION & ABUSE CHECK
- Evaluate the justification text. If it is too vague (like "general purchases") or looks like a personal expense hidden in a business category, flag it.

--------------------------------------------------
FINAL OUTPUT FORMAT:
Provide your final answer using clear text paragraphs. Do not output raw Python code, markdown code boxes, or programming lists. 

Format your response exactly like this:

### Audit Process Log:
- **Step 1 (Eligibility):** [Your observation here]
- **Step 2 (Client Validation):** [Your observation here]
- **Step 3 (Policy Thresholds):** [Your observation here]
- **Step 4 (Spending Evaluation):** [Your observation here]
- **Step 5 (Justification Check):** [Your observation here]

### Final Compliance Verdict:
- **Violation Severity:** [None, Low, Medium, or High]
- **Reason:** [A short, concise summary of your finding under 60 words, referencing the relevant policy sections]
"""

def extract_category_from_query(description: str) -> str:
    desc_lower = description.lower()
    if "meal" in desc_lower or "dinner" in desc_lower or "lunch" in desc_lower or "food" in desc_lower:
        return "Meals"
    if "party" in desc_lower or "client" in desc_lower or "entertain" in desc_lower:
        return "Entertainment"
    if "travel" in desc_lower or "flight" in desc_lower or "hotel" in desc_lower or "taxi" in desc_lower:
        return "Travel"
    if "software" in desc_lower or "license" in desc_lower or "saas" in desc_lower:
        return "Software"
    if "wifi" in desc_lower or "internet" in desc_lower:
        return "Wi-Fi"
    return "Meals"

def parse_clean_numbers(txn_data):
    """
    Silently calculates the values in the background.
    No programming jargon is returned to the user or shown on screen.
    """
    # 1. Parse amount
    amount_str = str(txn_data.get('amount', '0')).replace('$', '').replace('€', '').strip()
    try:
        amount_val = float(amount_str)
    except ValueError:
        amount_val = 0.0

    # 2. Parse headcount
    headcount_str = str(txn_data.get('headcount', '1'))
    nums = re.findall(r'\d+', headcount_str)
    headcount_val = int(nums[0]) if nums else 1

    # 3. Compute cost per person
    per_person_cost = round(amount_val / headcount_val, 2) if headcount_val > 0 else amount_val
    return amount_val, headcount_val, per_person_cost

def ask_llm(txn_data):
    # 1. Retrieve text from the uploaded PDF document
    document_text = get_document_text()

    category = txn_data.get("category") or extract_category_from_query(txn_data.get("description", ""))
    level = txn_data.get("level", "Junior")
    description = txn_data.get("description", "")

    # 2. Silently calculate numeric values using Python (reliable math!)
    amount_val, headcount_val, per_person_cost = parse_clean_numbers(txn_data)

    # 3. Query the Structure-Aware FAISS RAG to find matching sections
    if document_text and document_text.strip():
        get_or_create_retriever(document_text)
        policy_context = query_structure_aware_rag(
            query=description, 
            category=category, 
            level=level
        )
    else:
        policy_context = "Corporate compliance document context. Please refer to standard HE-FIN-004 thresholds."

    # 4. Construct the prompt with the math facts ALREADY calculated
    user_message = f"""
Please audit the following expense transaction data against the compliance rules:

TRANSACTION DATA:
- Employee Job Level: {level}
- Claimed Category: {category}
- Total Amount Charged: ${amount_val:.2f}
- Number of Attendees (Headcount): {headcount_val} person(s)
- Calculated Cost Per Person: ${per_person_cost:.2f}
- External Corporate Client: {txn_data.get('client', 'None')}
- Merchant Venue: {txn_data.get('merchant', 'Unknown')}
- Business Justification: "{description}"

POLICY TEXT EXCERPTS:
{policy_context}

Execute your 5-step evaluation now and provide the final verdict. Ensure your mathematical inequality comparisons in Step 4 are 100% logical.
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    return response["message"]["content"]