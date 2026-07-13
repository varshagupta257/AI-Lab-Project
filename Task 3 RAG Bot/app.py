import os
import re
from flask import Flask, render_template, request, jsonify, session
from chatbot.llm import ask_llm

app = Flask(__name__)
app.secret_key = "super_secret_audit_key_123"  # To sign sessions securely

# Mandatory fields needed before triggering the final RAG Audit
MANDATORY_FIELDS = {
    "level": "Employee Level (e.g., Junior, Senior, Director, VP)",
    "category": "Expense Category (e.g., Meals, Entertainment, Travel, Software)",
    "amount": "Total Amount of the transaction (e.g., $120)",
    "headcount": "Headcount / Total number of people (e.g., 1 person, or group of 3)",
    "client": "Client Field (Is there an external client present? If none, say 'None')",
    "merchant": "Merchant Venue (Where was this spent? e.g., Starbucks, Local Italian Restaurant)",
    "description": "Justification / Business purpose for this expense"
}

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()

    # 1. Initialize session storage if empty
    if "txn_data" not in session:
        session["txn_data"] = {field: None for field in MANDATORY_FIELDS}
        session["current_asking_field"] = None

    txn_data = session["txn_data"]
    current_asking = session["current_asking_field"]

    # 2. Save user response to the currently active question
    if current_asking:
        txn_data[current_asking] = user_message
        session["txn_data"] = txn_data
        session["current_asking_field"] = None
    else:
        # Heuristics for the initial message
        if not txn_data["description"]:
            txn_data["description"] = user_message
        
        lower_msg = user_message.lower()
        if "$" in user_message or "€" in user_message:
            amount_match = re.search(r'[\$\€](\d+(?:\.\d{2})?)', user_message)
            if amount_match:
                txn_data["amount"] = amount_match.group(1)
        
        if "meal" in lower_msg or "dinner" in lower_msg or "lunch" in lower_msg:
            txn_data["category"] = "Meals"
        elif "party" in lower_msg or "entertain" in lower_msg:
            txn_data["category"] = "Entertainment"
        elif "travel" in lower_msg or "flight" in lower_msg:
            txn_data["category"] = "Travel"

        session["txn_data"] = txn_data

    # 3. Find the next missing field
    next_missing_field = None
    for field in MANDATORY_FIELDS:
        if not session["txn_data"][field]:
            next_missing_field = field
            break

    # 4. If fields are missing, return a clean text response formatted for your UI
    if next_missing_field:
        session["current_asking_field"] = next_missing_field
        field_friendly_name = MANDATORY_FIELDS[next_missing_field]
        
        question_text = f"To audit this transaction accurately, I need a few more details. **Please provide the {field_friendly_name}.**"
        
        # Return structured JSON matching what your frontend script expects
        return jsonify({
            "status": "success",
            "message": question_text,
            "response": question_text
        })

    # 5. Run the final RAG compliance audit once all fields are collected
    final_audit_report = ask_llm(session["txn_data"])
    
    session.clear()  # Reset session state for the next audit

    return jsonify({
        "status": "success",
        "message": final_audit_report,
        "response": final_audit_report
    })

@app.route('/new_chat', methods=['POST'])
def new_chat():
    session.clear()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)