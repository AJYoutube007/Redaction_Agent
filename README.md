🧠 Redaction Agent Using UiPath SDK + LangChain

Tech Stack:
LangChain · UiPath Context Grounding · UiPath MCP Server · UiPath SDK · Python

🧾 Overview:
The Redaction Agent is an AI-powered automation bot built using LangChain and the UiPath Python SDK.
It intelligently scans unstructured documents — such as invoices, forms, or reports — to identify Personally Identifiable Information (PII) including:

🧍‍♂️ Names
📧 Emails
📱 Phone Numbers
💳 Account or ID Details, Bank Details

Once detected, the agent automatically redacts the PII from the document (PDF or text) by replacing it with masked values such as ██████ or [REDACTED].

⚙️ Workflow:

![Redaction_Agent_ToBeFlow](https://github.com/user-attachments/assets/567cf01e-6b38-4ccf-9c2f-37dc0e4da70b)

**Architecture:-** system overview in a single flowing paragraph

<img width="539" height="386" alt="Architecture_Diagram" src="https://github.com/user-attachments/assets/dceb7504-2c1e-483e-917f-637ba616beae" />

🚀 Key Features Summary		
		
Feature	Technology	Description

<img width="1026" height="904" alt="image" src="https://github.com/user-attachments/assets/83650395-25b0-4469-9e9b-28ad1236c46f" />

**Getting Started**

🧩 Getting Started
🔧 Prerequisites

Before you begin, ensure you have the following:

System Requirements:

✅ Python 3.10+

Access to UiPath Orchestrator
UiPath Orchestrator Setup

✅ Storage Buckets configured

✅ (Optional) Context Grounding Index set up

✅ API Credentials: Client ID, Client Secret, Tenant ID

✅ Enable Gemini API access & Generate API Key

✅ MCP Server (for Email Notifications)

✅ Email tool configured in MCP


**Installation**

1️⃣ Clone the Repository
```bash
git clone https://github.com/AJYoutube007/Redaction_Agent.git
cd Redaction_Agent
```

2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3️⃣ Set up environment variables
```bash
Create a .env file in the project root:
  # UiPath Orchestrator Configuration
   UIPATH_CLIENT_ID=your_client_id
   UIPATH_CLIENT_SECRET=your_client_secret
   UIPATH_TENANT_ID=your_tenant_id
   UIPATH_ORGANIZATION_ID=your_organization_id
   UIPATH_URL=https://cloud.uipath.com/
   
   # Google Gemini AI
   GOOGLE_API_KEY=your_gemini_api_key
   
   # MCP Server (for email notifications)
   MCP_SERVER_URL=your_mcp_server_url
```
4️⃣ Configure UiPath Storage Buckets
```bash
Create an input bucket (e.g., SB_LangChain)
Create an output bucket (e.g., SB_LangChain_Output)
Upload test PDFs to the input bucket
```

5️⃣.Set up Context Grounding Index (Optional)
```bash
Create a Context Grounding index named Redaction-Index
Upload your redaction policy document (DOCX/PDF format)
Ensure the document contains:

FOCUS ON / MUST REDACT: Values that should always be redacted
IGNORE / SAFE VALUES: Values that should never be redacted

Run index ingestion in Orchestrator
```
