import os
import json
import tempfile
import logging
import fitz  # PyMuPDF
import shutil
import atexit
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime

# --- UiPath SDK Import ---
from uipath import UiPath

# --- MCP Integration ---
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- Load Environment Variables ---
load_dotenv()

# --- Configure Logging ---
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pii_scanner.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GEMINI_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400

# Orchestrator Bucket Details
DEFAULT_BUCKET_KEY = "SB_LangChain" 
DEFAULT_OUTPUT_BUCKET_KEY = "SB_LangChain_Output"
DEFAULT_FOLDER_PATH = "Shared" 

# Context Grounding Configuration
REDACTION_INDEX_NAME = "Redaction-Index"
ENFORCE_CONTEXT_GROUNDING = True  # Set to True to make policy mandatory

# Email Configuration
DEFAULT_COMPLIANCE_EMAIL = os.getenv("COMPLIANCE_EMAIL", "AddyourEmail@gmail.com")
ENABLE_EMAIL_NOTIFICATIONS = os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "true").lower() == "true"

# Global variable for cleanup
TEMP_FILES_TO_CLEANUP = []

# --- Cleanup Function ---
def cleanup_temp_files():
    """Removes all temporary files created during processing."""
    global TEMP_FILES_TO_CLEANUP
    for path in TEMP_FILES_TO_CLEANUP:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {path}: {e}")

atexit.register(cleanup_temp_files)

# --- Enhanced PII Detection Prompt with Policy Context ---
ENHANCED_SENSITIVE_DATA_PROMPT = """
You are a highly specialized sensitive data analysis bot. Your primary task is to scan the provided text chunk for Personally Identifiable Information (PII) and other sensitive corporate data.

**REDACTION POLICY CONTEXT (from organizational guidelines):**
{policy_context}

**IMPORTANT: USE THE ABOVE POLICY TO GUIDE YOUR DETECTION:**
- Pay EXTRA attention to values mentioned in the "FOCUS ON" or "MUST REDACT" section
- DO NOT flag values mentioned in the "IGNORE" or "SAFE VALUES" section
- Apply the policy rules strictly while detecting PII below
- The policy takes precedence over general detection rules

Specific Sensitive Data Types to Look For (Be Strict):
1.  PII: Full Names (in context of being a person's name), Email Addresses, Phone Numbers (e.g., standard formats), Social Security Numbers (SSN), Passport Numbers, Dates of Birth, Home/Street Addresses.
2.  Financial: Credit Card Numbers (16-digit-like sequences), Bank Account/Routing Numbers.
3.  Confidential: Passwords (if explicit), API Keys (if explicit), or Internal Project Names (if contextually confidential).

Your Output Rules:
-   If sensitive data is found: Return a clean, concise, JSON list of all findings. You MUST include the full sensitive value in the "value" field for the redaction system.
    Example JSON Output (Mandatory Format):
    ```json
    [
      {{ "type": "Email Address", "value": "test@example.com"}},
      {{ "type": "Credit Card Number", "value": "4111-2222-3333-4444"}},
      {{ "type": "Full Name", "value": "John Doe"}}
    ]
    ```
-   If NO sensitive data is found: Return ONLY the following JSON object: {{"status": "No sensitive data found."}}

---
TEXT CHUNK TO ANALYZE:
{text}
---

Your Output (must be a valid JSON object or list):
"""

# -------------------- CONTEXT GROUNDING INTEGRATION --------------------

def extract_text(result):
    """Extract text content from Context Grounding result"""
    if hasattr(result, 'text'):
        return result.text.strip()
    if hasattr(result, 'content'):
        return result.content.strip()
    return str(result).strip()


def get_redaction_policy(sdk, folder_path: str = DEFAULT_FOLDER_PATH, folder_key: str = None):
    """
    Query the Redaction-Index for policy context using Context Grounding.
    Returns formatted policy text to be included in the prompt.
    
    If ENFORCE_CONTEXT_GROUNDING is True, raises exception if policy cannot be loaded.
    """
    try:
        print(f" Querying Context Grounding index: {REDACTION_INDEX_NAME}...")
        
        # Prepare folder parameters
        folder_params = {}
        if folder_key:
            folder_params["folder_key"] = folder_key
        elif folder_path:
            folder_params["folder_path"] = folder_path
        
        # GENERIC QUERIES 
        # Query 1: Get items to focus on/redact
        focus_results = sdk.context_grounding.search(
            name=REDACTION_INDEX_NAME,
            query="what should be redacted what to focus on must redact",
            number_of_results=20,
            **folder_params
        )
        
        # Query 2: Get items to ignore/skip
        ignore_results = sdk.context_grounding.search(
            name=REDACTION_INDEX_NAME,
            query="what should be ignored safe values do not redact skip",
            number_of_results=20,
            **folder_params
        )
        
        # Query 3: Get complete policy document
        policy_results = sdk.context_grounding.search(
            name=REDACTION_INDEX_NAME,
            query="redaction policy document guidelines rules",
            number_of_results=30,
            **folder_params
        )
        
        # Extract raw content from Context Grounding
        focus_content = []
        ignore_content = []
        
        # Collect all focus-related content
        for result in focus_results:
            text = extract_text(result)
            if text and len(text) > 5:  # Filter very short results
                focus_content.append(text)
        
        # Collect all ignore-related content
        for result in ignore_results:
            text = extract_text(result)
            if text and len(text) > 5:
                ignore_content.append(text)
        
        # Fallback: Use general policy results if specific queries are empty
        if not focus_content and not ignore_content and policy_results:
            print(" Using general policy content as fallback...")
            for result in policy_results:
                text = extract_text(result)
                if text and len(text) > 5:
                    # Simple heuristic: if contains certain keywords, categorize
                    lower_text = text.lower()
                    if any(word in lower_text for word in ['focus', 'redact', 'must', 'sensitive']):
                        focus_content.append(text)
                    elif any(word in lower_text for word in ['ignore', 'safe', 'skip', 'do not']):
                        ignore_content.append(text)
        
        # Remove duplicates while preserving order
        focus_content = list(dict.fromkeys(focus_content))
        ignore_content = list(dict.fromkeys(ignore_content))
        
        # ENFORCEMENT CHECK
        if ENFORCE_CONTEXT_GROUNDING:
            if not focus_content and not ignore_content:
                error_msg = f"""
 CONTEXT GROUNDING ENFORCEMENT FAILURE

The redaction policy could not be loaded from Context Grounding index '{REDACTION_INDEX_NAME}'.
This system is configured to REQUIRE a valid policy document for operation.

Possible causes:
1. Policy document not uploaded to the index
2. Index not ingested/synced recently
3. Policy document has incorrect format
4. Search queries cannot find policy content

ACTION REQUIRED:
- Verify '{REDACTION_INDEX_NAME}' exists in folder '{folder_path}'
- Ensure policy document (e.g., 'REDACTION POLICY.docx') is uploaded and ingested
- Re-ingest the index if document was recently updated
- Check index status in Orchestrator

To disable this enforcement, set ENFORCE_CONTEXT_GROUNDING = False in the code.
"""
                logger.error(error_msg)
                raise ValueError("Context Grounding policy enforcement failed. Cannot proceed without valid policy.")
        
        # Format policy context - COMPLETELY DRIVEN BY CONTEXT GROUNDING
        focus_text = "\n".join([f"{content}" for content in focus_content]) if focus_content else "No redaction rules provided by policy."
        ignore_text = "\n".join([f"{content}" for content in ignore_content]) if ignore_content else "No whitelist/ignore rules provided by policy."
        
        policy_text = f"""
========================================
REDACTION POLICY (from Context Grounding)
========================================

FOCUS ON - MUST REDACT:
{focus_text}

IGNORE - SAFE VALUES (DO NOT REDACT):
{ignore_text}

========================================
INSTRUCTIONS FOR AI SCANNER:
- Carefully read the policy sections above
- Apply the rules EXACTLY as specified in the policy document
- For values listed in FOCUS section: ALWAYS flag for redaction
- For values listed in IGNORE section: NEVER flag for redaction
- The policy takes absolute precedence over general detection rules
========================================
"""
        
        print(f"✓ Policy context loaded from {REDACTION_INDEX_NAME}")
        print(f"  - Loaded {len(focus_content)} focus policy items")
        print(f"  - Loaded {len(ignore_content)} ignore policy items")
        
        # MINIMUM CONTENT CHECK
        if ENFORCE_CONTEXT_GROUNDING:
            total_content = len(focus_content) + len(ignore_content)
            if total_content < 2:
                logger.warning(f" Only {total_content} policy items found. Policy may be incomplete or not fully ingested.")
                logger.warning("Continuing with available rules, but results may not follow complete policy.")
        
        return policy_text
        
    except Exception as e:
        logger.error(f"Failed to load redaction policy from Context Grounding: {e}")
        
        if ENFORCE_CONTEXT_GROUNDING:
            logger.error(" Context Grounding enforcement is enabled. Cannot proceed without policy.")
            raise ValueError(f"Context Grounding policy enforcement failed: {e}")
        else:
            logger.warning(" Falling back to standard detection without custom policy")
            return "No custom policy loaded. Using standard PII detection."

# -------------------- MCP SERVER INTEGRATION --------------------

@asynccontextmanager
async def get_mcp_session():
    """MCP session management for external tool integration"""
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
    
    if not MCP_SERVER_URL:
        logger.error("MCP_SERVER_URL not found in environment variables")
        raise ValueError("MCP_SERVER_URL must be set in .env file")
    
    uipath_client = UiPath()
    UIPATH_ACCESS_TOKEN = None
    
    try:
        if hasattr(uipath_client, 'api_client') and hasattr(uipath_client.api_client, 'default_headers'):
            auth_header = uipath_client.api_client.default_headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                UIPATH_ACCESS_TOKEN = auth_header.replace('Bearer ', '')
    except Exception as e:
        logger.warning(f"Could not extract UiPath token: {e}")
    
    try:
        async with streamablehttp_client(
            url=MCP_SERVER_URL,
            headers={"Authorization": f"Bearer {UIPATH_ACCESS_TOKEN}"} if UIPATH_ACCESS_TOKEN else {},
            timeout=60,
        ) as (read, write, session_id_callback):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except Exception as e:
        logger.error(f"Failed to create MCP session: {e}")
        raise


async def send_email_mcp(recipient: str, subject: str, body: str):
    """Send email using MCP server email tool"""
    try:
        async with get_mcp_session() as session:
            tools = await load_mcp_tools(session)
            
            email_tool = None
            for tool in tools:
                tool_name = tool.name
                if any(keyword in tool_name.lower() for keyword in ["send_mail_rpa_mcp"]):
                    email_tool = tool
                    break
            
            if not email_tool:
                logger.error("Email tool not found in MCP server")
                raise Exception("Email tool not available in MCP server")
            
            params = {
                "In_Receipant": recipient,
                "In_Subject": subject,
                "In_Body": body
            }
            
            await email_tool.ainvoke(params)
            print(f"✓ Email sent to {recipient}")
            
    except Exception as e:
        logger.error(f"Failed to send email via MCP: {e}", exc_info=True)
        raise


async def send_pii_scan_report_email(findings_data: dict, recipient: str = None, 
                                     redacted_pdf_path: str = None, bucket_name: str = None,
                                     policy_used: bool = False):
    """Send formatted PII scan report via MCP email"""
    if not ENABLE_EMAIL_NOTIFICATIONS:
        return
    
    recipient = recipient or DEFAULT_COMPLIANCE_EMAIL
    
    file_name = findings_data.get('file', 'Unknown')
    total_findings = findings_data.get('total_findings', 0)
    findings = findings_data.get('findings', [])
    bucket = findings_data.get('bucket', 'Unknown')
    
    high_risk_types = {'SSN', 'Social Security Number', 'Credit Card Number', 'Passport Number', 'Bank Account'}
    has_high_risk = any(f['data_type'] in high_risk_types for f in findings)
    severity = "[HIGH RISK]" if has_high_risk else "[STANDARD]"
    
    # Policy badge
    policy_badge = ""
    if policy_used:
        policy_badge = """
        <div style="display: inline-block; background-color: #4caf50; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; margin-left: 10px;">
            ✓ Context Grounding Policy Applied
        </div>
        """
    
    # Limit findings to first 5 for email (avoid size limit)
    findings_to_show = findings[:5] if findings else []
    has_more_findings = len(findings) > 5
    
    findings_html = ""
    if findings_to_show:
        for idx, finding in enumerate(findings_to_show, 1):
            findings_html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{idx}</td>
                <td style="padding: 8px; border: 1px solid #ddd;"><b>{finding['data_type']}</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;"><code>{finding['masked_value']}</code></td>
                <td style="padding: 8px; border: 1px solid #ddd;">Page {finding['source_page']}</td>
            </tr>
            """
        
        # Add "more findings" row if truncated
        if has_more_findings:
            findings_html += f"""
            <tr style="background-color: #fff3e0;">
                <td colspan="4" style="padding: 8px; border: 1px solid #ddd; text-align: center; font-weight: bold;">
                    + {len(findings) - 5} more findings (see full report in logs/output)
                </td>
            </tr>
            """
    elif total_findings == 0:
        findings_html = """
        <tr>
            <td colspan="4" style="padding: 8px; border: 1px solid #ddd; text-align: center;">
                No sensitive data detected
            </td>
        </tr>
        """
    
    # Download instructions section
    download_section = ""
    if redacted_pdf_path and bucket_name:
        download_section = f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #4caf50;">
            <h3 style="margin-top: 0; color: #2e7d32;"> Download Redacted PDF</h3>
            <p><b>Bucket:</b> <code>{bucket_name}</code></p>
            <p><b>File Path:</b> <code>{redacted_pdf_path}</code></p>
            <p><b>Folder:</b> <code>{DEFAULT_FOLDER_PATH}</code></p>
            <p style="margin-bottom: 0;">
                <b>Access Instructions:</b><br/>
                1. Log into UiPath Orchestrator<br/>
                2. Navigate to <b>Storage Buckets</b><br/>
                3. Open bucket: <b>{bucket_name}</b><br/>
                4. Download file: <b>{redacted_pdf_path}</b>
            </p>
        </div>
        """

    subject = f"Redaction & Sensitive Data Alert {severity} - {file_name}"
    body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: #{"d32f2f" if has_high_risk else "f57c00"}; color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background-color: #1976d2; color: white; padding: 10px; text-align: left; }}
            .footer {{ background-color: #f5f5f5; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1> Sensitive Data & Redaction Report</h1>
            <p style="margin: 0;">Automated Sensitive Data Detection & Redaction{policy_badge}</p>
        </div>
        
        <div class="content">
            <div class="summary">
                <h2 style="margin-top: 0;"> Scan Summary</h2>
                <p><b>Document:</b> {file_name}</p>
                <p><b>Source Bucket:</b> {bucket}</p>
                <p><b>Scan Time:</b> {datetime.now().isoformat()}</p>
                <p><b>Total Findings:</b> <span style="font-size: 24px; color: #{"d32f2f" if total_findings > 0 else "4caf50"};">{total_findings}</span></p>
                <p><b>Risk Level:</b> {severity}</p>
                {"<p><b>Policy Engine:</b> ✓ Context Grounding (Redaction-Index)</p>" if policy_used else ""}
            </div>
            
            <h3> Detected & Redacted Sensitive Data</h3>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Data Type</th>
                        <th>Masked Value</th>
                        <th>Location</th>
                    </tr>
                </thead>
                <tbody>
                    {findings_html}
                </tbody>
            </table>
            
            {"<p style='color: #d32f2f; font-weight: bold;'> ACTION REQUIRED: This document contained high-risk PII which has been redacted.</p>" if has_high_risk else ""}
            
            {download_section}
            
            <div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ff9800;">
                <h3 style="margin-top: 0;"> Next Steps</h3>
                <ul style="margin-bottom: 0;">
                    <li>Download the redacted PDF from the storage bucket using the instructions above</li>
                    <li>All sensitive information has been permanently redacted with black boxes</li>
                    <li>Review the redacted document before distribution</li>
                    <li>Update data handling procedures if needed</li>
                    <li>Keep this email for compliance audit trail</li>
                </ul>
            </div>
            
            <p style="background-color: #e3f2fd; padding: 10px; border-radius: 5px;">
                <b> Note:</b> The redacted PDF has been uploaded to the output bucket. 
                All PII detected by AI has been permanently removed from the document.
                The original file remains unchanged in the source bucket.
            </p>
        </div>
        
        <div class="footer">
            <p>This is an automated report from the PII Scanner & Redaction System</p>
            <p>Powered by UiPath Automation + Google Gemini AI + Context Grounding + PyMuPDF</p>
            <p style="color: #999; margin-top: 10px;">Generated: {datetime.now().isoformat()}</p>
        </div>
    </body>
    </html>
    """
    
    try:
        await send_email_mcp(recipient, subject, body)
    except Exception as e:
        logger.error(f"Email notification failed: {e}")


# -------------------- PDF REDACTION FUNCTIONS --------------------

def redact_and_save(input_pdf_path, output_pdf_path, search_string, redact_text="REDACTED"):
    """
    Finds and permanently redacts ALL occurrences of a specific string in input_pdf_path 
    and saves the modified document to output_pdf_path.
    """
    try:
        doc = fitz.open(input_pdf_path)
    except Exception as e:
        logger.error(f"Error opening PDF for redaction: {e}")
        return 0

    redaction_count = 0

    for page in doc:
        search_results = page.search_for(search_string)
        page_redactions = 0
        
        for rect in search_results:
            page.add_redact_annot(rect, text=redact_text, fill=(0, 0, 0), fontsize=8, text_color=(1, 1, 1))
            page_redactions += 1

        if page_redactions > 0:
            page.apply_redactions()
            redaction_count += page_redactions

    doc.save(output_pdf_path, garbage=4, deflate=True) 
    doc.close()
    
    return redaction_count


def perform_pdf_redaction(local_pdf_path: str, pii_strings: set) -> str:
    """
    Performs ping-pong redaction on a PDF using PyMuPDF.
    Returns the path to the final redacted PDF.
    """
    global TEMP_FILES_TO_CLEANUP
    
    if not pii_strings:
        return local_pdf_path
    
    print(f"Starting redaction of {len(pii_strings)} PII strings...")
    
    try:
        temp_dir = os.path.dirname(local_pdf_path)
        
        # Create two temporary files for ping-pong
        with tempfile.NamedTemporaryFile(suffix="_a.pdf", dir=temp_dir, delete=False) as tmp_a:
            temp_file_a = tmp_a.name
        with tempfile.NamedTemporaryFile(suffix="_b.pdf", dir=temp_dir, delete=False) as tmp_b:
            temp_file_b = tmp_b.name
        
        TEMP_FILES_TO_CLEANUP.extend([temp_file_a, temp_file_b])
        
        # Initialize: Copy original PDF to starting file
        shutil.copy(local_pdf_path, temp_file_a)
        
        current_input = temp_file_a
        current_output = temp_file_b
        total_redactions = 0
        
        # Redaction loop: Ping-pong between files
        for pii_string in pii_strings:
            redacted_count = redact_and_save(
                input_pdf_path=current_input,
                output_pdf_path=current_output,
                search_string=pii_string,
                redact_text="REDACTED"
            )
            
            total_redactions += redacted_count
            
            # Swap files for next iteration
            current_input, current_output = current_output, current_input
        
        # Final redacted file is in current_input (due to last swap)
        final_redacted_path = current_input
        
        print(f"✓ Redaction complete: {total_redactions} instances redacted")
        
        return final_redacted_path
        
    except Exception as e:
        logger.error(f"PDF redaction failed: {e}", exc_info=True)
        raise


# -------------------- CORE SCANNER FUNCTION --------------------

async def run_sensitive_data_scan(bucket_key: str, file_path: str, folder_path: str = DEFAULT_FOLDER_PATH, 
                                   folder_key: str = None, output_bucket_key: str = None, 
                                   email_recipient: str = None):
    """
    Main function to scan PDF for sensitive data, redact it, and upload to bucket
    Enhanced with Context Grounding policy integration
    """
    print(f"\n{'='*70}")
    print(f" Starting PII Scan with Context Grounding: {file_path}")
    print(f"{'='*70}\n")
    
    local_pdf_path = None
    final_report = []
    policy_used = False

    # 1. Initialize UiPath SDK
    sdk = UiPath()

    # 2. Load Redaction Policy from Context Grounding
    policy_context = get_redaction_policy(sdk, folder_path, folder_key)
    policy_used = "No custom policy loaded" not in policy_context

    # 3. Initialize Model
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    except ValueError as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # 4. Storage Bucket Download
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.basename(file_path), delete=False) as temp_file:
            local_pdf_path = temp_file.name
        
        TEMP_FILES_TO_CLEANUP.append(local_pdf_path)
        
        download_params = {
            "name": bucket_key,
            "blob_file_path": file_path,
            "destination_path": local_pdf_path
        }
        
        if folder_key:
            download_params["folder_key"] = folder_key
        elif folder_path:
            download_params["folder_path"] = folder_path
            
        sdk.buckets.download(**download_params)
        print(f"✓ Downloaded: {os.path.getsize(local_pdf_path) / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"File download failed: {e}", exc_info=True)
        return

    # 5. Load and Split the Document
    try:
        loader = PyPDFLoader(local_pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Loaded {len(documents)} pages, split into {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"PDF loading/splitting failed: {e}", exc_info=True)
        return

    # 6. Run LangChain Detection with Policy Context
    prompt = ChatPromptTemplate.from_template(ENHANCED_SENSITIVE_DATA_PROMPT)
    detection_chain = prompt | llm

    print(f"\n Scanning {len(chunks)} chunks for PII (with policy guidance)...")

    unique_pii_strings = set()  # For redaction
    
    for i, chunk in enumerate(chunks):
        try:
            page_info = chunk.metadata.get("page", f"Chunk {i+1}") 
            
            # Progress indicator
            print(f" Processing chunk {i+1}/{len(chunks)} (Page {page_info})...", end=" ")
            
            # Invoke with both text and policy context
            response = detection_chain.invoke({
                "text": chunk.page_content,
                "policy_context": policy_context
            })
            
            json_string = response.content.strip()
            if json_string.startswith("```json"):
                json_string = json_string[7:].strip()
            if json_string.endswith("```"):
                json_string = json_string[:-3].strip()

            findings = json.loads(json_string)
            
            if isinstance(findings, list) and findings:
                print(f"✓ Found {len(findings)} PII item(s)")
                for finding in findings:
                    # Add to unique PII set for redaction
                    if "value" in finding:
                        unique_pii_strings.add(finding["value"].strip())
                    
                    # Add to report with masked value
                    final_report.append({
                        "data_type": finding.get("type", "Unknown"),
                        "masked_value": finding.get("value", "N/A")[:5] + "..." if finding.get("value") else "N/A",
                        "source_page": page_info,
                        "source_chunk_index": i,
                        "file_path": file_path
                    })
            else:
                print("✓ Clean")
            
        except json.JSONDecodeError:
            print(" Parse error")
            logger.warning(f"Chunk {i+1} returned malformed data")
        except Exception as e:
            print(f" Error")
            logger.error(f"Error on chunk {i+1}: {e}")

    # 7. Perform PDF Redaction
    redacted_pdf_path = None
    redacted_filename = None
    
    if unique_pii_strings:
        print(f"\n Found {len(unique_pii_strings)} unique PII values")
        
        try:
            # Perform redaction
            redacted_temp_path = perform_pdf_redaction(local_pdf_path, unique_pii_strings)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            redacted_filename = f"{base_name}_redacted.pdf"
            
            # Upload redacted PDF to output bucket
            if output_bucket_key:
                upload_params = {
                    "name": output_bucket_key,
                    "blob_file_path": redacted_filename,
                    "source_path": redacted_temp_path
                }
                
                if folder_key:
                    upload_params["folder_key"] = folder_key
                elif folder_path:
                    upload_params["folder_path"] = folder_path
                
                sdk.buckets.upload(**upload_params)
                print(f"✓ Uploaded redacted PDF: {redacted_filename}")
                redacted_pdf_path = redacted_filename
            
        except Exception as e:
            logger.error(f"PDF redaction/upload failed: {e}", exc_info=True)
    else:
        print("\n✓ No PII found - no redaction needed")

    # 8. Generate Report and Send Email
    print(f"\n{'='*70}")
    print(f"SCAN COMPLETE: {len(final_report)} findings")
    if policy_used:
        print(f"✓ Context Grounding Policy Applied")
    print(f"{'='*70}\n")
    
    if final_report:
        for idx, finding in enumerate(final_report, 1):
            print(f"  {idx}. {finding['data_type']}: {finding['masked_value']} (Page: {finding['source_page']})")
    
    # Prepare report data
    report_data = {
        "file": file_path,
        "bucket": bucket_key,
        "scan_timestamp": datetime.now().isoformat(),
        "total_chunks_analyzed": len(chunks),
        "total_findings": len(final_report),
        "findings": final_report,
        "redacted_pdf": redacted_pdf_path,
        "output_bucket": output_bucket_key,
        "policy_applied": policy_used
    }
    
    # Send email notification
    if ENABLE_EMAIL_NOTIFICATIONS:
        try:
            await send_pii_scan_report_email(
                report_data, 
                email_recipient, 
                redacted_pdf_path, 
                output_bucket_key,
                policy_used
            )
        except Exception as e:
            logger.error(f"Email notification failed: {e}")

    return final_report, redacted_pdf_path

# -------------------- UIPATH ENTRY POINT --------------------

async def main(input_dict):
    """
    UiPath-compatible entry point with async support
    Accepts a single dict from -f input.json
    """
    bucket_key = input_dict.get("bucket_key", DEFAULT_BUCKET_KEY)
    file_path  = input_dict.get("file_path")
    folder_path = input_dict.get("folder_path", DEFAULT_FOLDER_PATH)
    folder_key = input_dict.get("folder_key")
    output_bucket_key = input_dict.get("output_bucket_key", DEFAULT_OUTPUT_BUCKET_KEY)
    email_recipient = input_dict.get("email_recipient")

    if not file_path:
        error_msg = "'file_path' argument is mandatory"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}

    try:
        results, redacted_pdf = await run_sensitive_data_scan(
            bucket_key, 
            file_path, 
            folder_path, 
            folder_key, 
            output_bucket_key,
            email_recipient
        )

        output = {
            "status": "completed",
            "file_scanned": file_path,
            "bucket_used": bucket_key,
            "total_findings": len(results) if results else 0,
            "redacted_pdf": redacted_pdf,
            "output_bucket": output_bucket_key,
            "results": results,
            "context_grounding_used": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return output
        
    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "file_scanned": file_path
        }

# -------------------- LOCAL TEST --------------------

if __name__ == "__main__":
    import asyncio
    
    TEST_FILE_NAME = "Redaction_Test.pdf"
    
    final_output = asyncio.run(main({
        "bucket_key": DEFAULT_BUCKET_KEY,
        "file_path": TEST_FILE_NAME,
        "folder_path": DEFAULT_FOLDER_PATH,
        "output_bucket_key": DEFAULT_OUTPUT_BUCKET_KEY,
        # "email_recipient": "test@example.com"  # Optional: override default
    }))
    
    print("\n" + "="*70)
    print("FINAL OUTPUT")
    print("="*70)
    print(json.dumps(final_output, indent=2))
