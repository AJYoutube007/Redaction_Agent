Redaction Agent Using UiPath SDK + LangChain 

**Tech Stack- LangChain, UiPath Context Grounding, UiPath MCP Server,UiPath SDK, Python**

The Redaction Agent is an AI-powered automation bot built using LangChain and the UiPath Python SDK.
It intelligently scans documents (such as invoices, forms, or reports), identifies Personally Identifiable Information (PII) like names, emails, phone numbers, and account details, and automatically redacts

Replaces detected PII in PDF or text documents with masked values (e.g., ██████ or [REDACTED]).

**Workflow-** 

![Redaction_Agent_ToBeFlow](https://github.com/user-attachments/assets/567cf01e-6b38-4ccf-9c2f-37dc0e4da70b)

**Architecture:-** system overview in a single flowing paragraph

<img width="539" height="386" alt="Architecture_Diagram" src="https://github.com/user-attachments/assets/dceb7504-2c1e-483e-917f-637ba616beae" />

**Key Features Summary**		
		
Feature	Technology	Description

<img width="1026" height="904" alt="image" src="https://github.com/user-attachments/assets/83650395-25b0-4469-9e9b-28ad1236c46f" />

**Getting Started**

Prerequisites:

Before you begin, ensure you have the following:

  Python 3.8+ installed on your system
  
UiPath Orchestrator account with:
  Storage Buckets configured
  Context Grounding index set up (optional but recommended)
  Valid API credentials (Client ID, Client Secret, Tenant ID)

Google Cloud account with:

  Gemini API access enabled
  API key generated

MCP Server (for using email notifications)

  UiPath MCP server URL
  Email tool configured

**Installation**

Clone the repository

