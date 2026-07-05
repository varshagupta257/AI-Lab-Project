import pandas as pd
import requests
import json
import re
import os

# ==========================================
# Configuration & Absolute Paths
# ==========================================
INPUT_FILE = '/Users/varshagupta/Desktop/VS Code/AI Lab/Project/data/expense_transactions-2.csv'
OUTPUT_FILE = '/Users/varshagupta/Desktop/VS Code/AI Lab/Project/Audited_Expenses_Output.csv'
OLLAMA_API_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3.2:3b'

# ==========================================
# Simplified Prompt 1: Zero-Shot
# ==========================================
ZERO_SHOT_PROMPT = (
    "You are a corporate expense auditor. Read this transaction and perform three tasks:\n"
    "1. Categorize the expense. Choose ONLY from: [Meals, Travel, Software, Office Supplies, Entertainment].\n"
    "2. Decide the violation level (None, Low, Medium, High) based on whether the amount makes sense for the justification.\n"
    "3. Summarize the justification in a professional phrase that is STRICTLY 5 to 8 words long.\n\n"
    "Output ONLY a valid JSON object with keys: \"Extracted_Category\", \"Violation_Level\", \"Summary\".\n\n"
    "Amount: {amount}\n"
    "Justification: {justification}"
)

# ==========================================
# Simplified Prompt 2: Few-Shot CoT (4 Examples)
# ==========================================
FEW_SHOT_COT_PROMPT = (
    "You are a corporate expense auditor. Review the transaction, categorize it into one of the allowed categories "
    "[Meals, Travel, Software, Office Supplies, Entertainment], think step-by-step to assess compliance, and output a strict JSON format.\n\n"
    
    "Example 1 (Normal Expense - None):\n"
    "Amount: 45.00\n"
    "Justification: Team lunch meeting.\n"
    "Thought Process:\n"
    "1. A team lunch fits perfectly under the 'Meals' category.\n"
    "2. The amount of $45.00 is standard for a business team lunch. No policies are broken.\n"
    "3. Verdict: Violation level is None.\n"
    "JSON Output:\n"
    "{{\n"
    "  \"Extracted_Category\": \"Meals\",\n"
    "  \"Violation_Level\": \"None\",\n"
    "  \"Summary\": \"Standard corporate team lunch meeting\"\n"
    "}}\n\n"

    "Example 2 (Minor Issue - Low):\n"
    "Amount: 85.00\n"
    "Justification: Taxi to airport during surge pricing.\n"
    "Thought Process:\n"
    "1. An airport taxi ride belongs under the 'Travel' category.\n"
    "2. $85.00 is slightly high for a taxi ride, but the surge pricing explanation is acceptable though needs monitoring.\n"
    "3. Verdict: Violation level is Low.\n"
    "JSON Output:\n"
    "{{\n"
    "  \"Extracted_Category\": \"Travel\",\n"
    "  \"Violation_Level\": \"Low\",\n"
    "  \"Summary\": \"Airport taxi transfer with surge pricing\"\n"
    "}}\n\n"

    "Example 3 (Missing Info - Medium):\n"
    "Amount: 320.00\n"
    "Justification: Monthly project tool subscription renewal.\n"
    "Thought Process:\n"
    "1. A project tool subscription falls under the 'Software' category.\n"
    "2. $320.00 is a significant recurring monthly cost, but the justification fails to name the specific tool.\n"
    "3. Verdict: Violation level is Medium.\n"
    "JSON Output:\n"
    "{{\n"
    "  \"Extracted_Category\": \"Software\",\n"
    "  \"Violation_Level\": \"Medium\",\n"
    "  \"Summary\": \"Unspecified monthly project tool software subscription\"\n"
    "}}\n\n"

    "Example 4 (Severe Violation - High):\n"
    "Amount: 850.00\n"
    "Justification: Dinner.\n"
    "Thought Process:\n"
    "1. Dinner fits under the 'Meals' category.\n"
    "2. $850.00 is excessively high for a single dinner, and the justification provides zero business context or client details.\n"
    "3. Verdict: Violation level is High.\n"
    "JSON Output:\n"
    "{{\n"
    "  \"Extracted_Category\": \"Meals\",\n"
    "  \"Violation_Level\": \"High\",\n"
    "  \"Summary\": \"Excessive dinner cost lacking business context\"\n"
    "}}\n\n"

    "Now evaluate the following transaction fairly using the allowed categories [Meals, Travel, Software, Office Supplies, Entertainment]:\n"
    "Amount: {amount}\n"
    "Justification: {justification}\n\n"
    "Ensure your final \"Summary\" string in the JSON is STRICTLY between 5 and 8 words long."
)

# ==========================================
# Core Logic
# ==========================================
def call_ollama(prompt_template, row):
    amount = row.get('Amount', '0.00')
    justification = row.get('Justification', 'No justification provided.')
    
    prompt = prompt_template.format(
        amount=amount,
        justification=justification
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "stop": ["}\n", "}]"]
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=45)
        res_text = response.json().get('response', '')
        if "{" in res_text and "}" not in res_text:
            res_text += "}"
        return res_text
    except Exception:
        return ""

def extract_json(response_text):
    try:
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {"Extracted_Category": "Error", "Violation_Level": "Error", "Summary": "Failed to parse."}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Cannot find file at {INPUT_FILE}")
        return

    # CRITICAL FIX: Tell pandas to split the columns by semicolon!
    df = pd.read_csv(INPUT_FILE, sep=';')
    
    # Clean the column headers just to be perfectly safe
    df.columns = df.columns.str.strip()
    
    total_rows = len(df)
    print(f"Processing ALL {total_rows} transactions using correct semicolon delimiter...")
    
    for index, row in df.iterrows():
        # Now that the columns are actually split, it will find Transaction_ID effortlessly
        txn_id = str(row.get('Transaction_ID', f"TXN-{1001 + int(index)}"))

        print(f"[{int(index) + 1}/{total_rows}] Auditing {txn_id}...")
        
        zs_resp = call_ollama(ZERO_SHOT_PROMPT, row)
        zs_data = extract_json(zs_resp)
        
        fs_resp = call_ollama(FEW_SHOT_COT_PROMPT, row)
        fs_data = extract_json(fs_resp)
        
        row_result = pd.DataFrame([{
            "Transaction_ID": txn_id,
            "Date": row.get('Date', 'N/A'),
            "Amount": row.get('Amount', 'N/A'),
            "ZeroShot_Category": zs_data.get("Extracted_Category", "N/A"),
            "ZeroShot_Violation": zs_data.get("Violation_Level", "N/A"),
            "ZeroShot_Summary": zs_data.get("Summary", "N/A"),
            "FewShot_Category": fs_data.get("Extracted_Category", "N/A"),
            "FewShot_Violation": fs_data.get("Violation_Level", "N/A"),
            "FewShot_Summary": fs_data.get("Summary", "N/A")
        }])
        
        if int(index) == 0:
            row_result.to_csv(OUTPUT_FILE, index=False, mode='w')
        else:
            row_result.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

    print(f"\nSUCCESS! File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()