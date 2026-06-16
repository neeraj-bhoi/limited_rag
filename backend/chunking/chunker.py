import re

# Multilingual labels for converting JSON service details into semantic Markdown
LABELS = {
    "en": {
        "title": "Service Profile",
        "sno": "SNo",
        "service_id": "Service ID",
        "dept": "Department",
        "category": "Category",
        "sla": "SLA",
        "time_limit": "Time Limit",
        "contact": "Contact Details",
        "fees_section": "Fee and Application Details",
        "kiosk_fee": "Kiosk/Center Fee",
        "online_fee": "Online Application Fee",
        "where_to_apply": "Where to Apply",
        "fee_info": "Fee Information details",
        "docs_section": "Required Documents",
        "doc_type": "Document Type Category",
        "sup_doc": "Supporting Document Name",
        "mandatory": "Mandatory",
        "fields_section": "Form Input Fields (to be filled during online application)",
        "field_label": "Field Label",
        "input_type": "Input Type",
        "data_type": "Data Type"
    },
    "hi": {
        "title": "सेवा प्रोफ़ाइल",
        "sno": "क्रमांक",
        "service_id": "सेवा आईडी (Service ID)",
        "dept": "विभाग (Department)",
        "category": "श्रेणी (Category)",
        "sla": "एसएलए (SLA)",
        "time_limit": "समय सीमा (Time Limit)",
        "contact": "संपर्क विवरण (Contact Details)",
        "fees_section": "शुल्क और आवेदन विवरण (Fee & Application)",
        "kiosk_fee": "कियोस्क/केंद्र शुल्क (Kiosk Fee)",
        "online_fee": "ऑनलाइन आवेदन शुल्क (Online Fee)",
        "where_to_apply": "कहाँ आवेदन करें (Where to Apply)",
        "fee_info": "शुल्क संबंधी जानकारी (Fee Info)",
        "docs_section": "आवश्यक दस्तावेज़ (Required Documents)",
        "doc_type": "दस्तावेज़ प्रकार श्रेणी",
        "sup_doc": "सहायक दस्तावेज़ का नाम",
        "mandatory": "अनिवार्य",
        "fields_section": "फॉर्म इनपुट फील्ड (ऑनलाइन आवेदन के दौरान भरे जाने वाले)",
        "field_label": "फ़ील्ड नाम (Label)",
        "input_type": "इनपुट प्रकार (Input Type)",
        "data_type": "डेटा प्रकार (Data Type)"
    }
}

def clean_text(text):
    """
    Cleans raw text by removing excessive white spaces and blank lines.
    """
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=1500, overlap=200):
    """
    Splits text into sliding-window chunks with a set overlap.
    """
    if not text:
        return []
    
    # Clean up excess spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    text_len = len(text)
    chunks = []
    
    start = 0
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (chunk_size - overlap)
        
    return [c for c in chunks if c]

def convert_json_to_markdown(json_data, lang="en", exclude_form_fields=False):
    """
    Converts a structured service detail JSON profile into a semantic, LLM-friendly Markdown document.
    Handles the nested required documents and form field parameters.
    """
    lbl = LABELS.get(lang, LABELS["en"])
    
    sno = json_data.get("sno", "")
    service_id = json_data.get("service_id", "")
    name = json_data.get("name", "")
    department = json_data.get("department", "")
    category = json_data.get("category", "")
    sla = json_data.get("sla", "")
    time_limit = json_data.get("time_limit", "")
    contact = json_data.get("contact_details", "")
    
    md = []
    md.append(f"# {lbl['title']}: {name}")
    md.append(f"- **{lbl['sno']}**: {sno}")
    if service_id:
        md.append(f"- **{lbl['service_id']}**: {service_id}")
    if department:
        md.append(f"- **{lbl['dept']}**: {department}")
    if category:
        md.append(f"- **{lbl['category']}**: {category}")
    if sla:
        md.append(f"- **{lbl['sla']}**: {sla}")
    if time_limit:
        md.append(f"- **{lbl['time_limit']}**: {time_limit}")
    if contact:
        md.append(f"- **{lbl['contact']}**: {contact}")
    
    # Fees and Application
    fees = json_data.get("fees", {})
    kiosk_fee = fees.get("kiosk_fee", "")
    online_fee = fees.get("online_fee", "")
    where_to_apply = fees.get("where_to_apply", "")
    raw_text = fees.get("raw_text", "")
    
    md.append(f"\n## {lbl['fees_section']}")
    if kiosk_fee:
        md.append(f"- **{lbl['kiosk_fee']}**: ₹{kiosk_fee}")
    if online_fee:
        md.append(f"- **{lbl['online_fee']}**: ₹{online_fee}")
    if where_to_apply:
        md.append(f"- **{lbl['where_to_apply']}**: {where_to_apply}")
    if raw_text:
        md.append(f"- **{lbl['fee_info']}**: {clean_text(raw_text)}")
        
    # Required Documents (updated for nested structures)
    docs = json_data.get("required_documents", [])
    if docs:
        md.append(f"\n## {lbl['docs_section']}")
        for d in docs:
            doc_type = d.get("document_type", "").strip()
            mandatory = d.get("mandatory", "").strip()
            sno_val = d.get("sno", "")
            
            md.append(f"- Category [{sno_val}]: {doc_type} (Mandatory/अनिवार्य: {mandatory})")
            
            # Sub-documents list
            sub_docs = d.get("supporting_documents", [])
            for sub_idx, sub in enumerate(sub_docs):
                sub_name = sub.get("name", "").strip()
                pdf_link = sub.get("format_link", "").strip()
                link_str = f" (Format/प्रारूप: {pdf_link})" if pdf_link else ""
                md.append(f"  * Option {sno_val}.{sub_idx+1}: {sub_name}{link_str}")
                
    # Form input fields
    fields = json_data.get("form_fields", [])
    if fields and not exclude_form_fields:
        md.append(f"\n## {lbl['fields_section']}")
        for f in fields:
            label = f.get("label", "").strip()
            input_type = f.get("input_type", "").strip()
            data_type = f.get("data_type", "").strip()
            if label:
                md.append(f"- {lbl['field_label']}: {label} ({lbl['input_type']}: {input_type}, {lbl['data_type']}: {data_type})")
                
    return "\n".join(md)
