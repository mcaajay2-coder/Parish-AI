import re
import frappe
from rapidfuzz import fuzz

RESERVED_GENERIC_WORDS = {
    'any', 'some', 'all', 'few', 'many', 'member', 'members', 'parish', 'parishioner',
    'parishioners', 'family', 'families', 'details', 'detail', 'info', 'information',
    'records', 'record', 'list', 'show', 'give', 'get', 'find', 'total', 'count',
    'number', 'people', 'persons', 'person', 'sacrament', 'sacraments', 'sacrement',
    'sacrements', 'baptism', 'baptisms', 'confirmation', 'communion', 'marriage',
    'death', 'my', 'the', 'a', 'an', 'in', 'of', 'for', 'about', 'on', 'to', 'please',
    'can', 'you', 'tell', 'me', 'registered', 'whose', 'names', 'start', 'with',
    'starting', 'begins', 'parishes', 'diocese', 'vicariate', 'status', 'certificate'
}

def normalize_name(name_str: str) -> str:
    """
    Normalizes a name for comparison:
    - Lowercase
    - Trim spaces
    - Collapse multiple spaces into one
    - Remove punctuation (.,-_/\\\'"), normalize '.' and other separators
    - Preserve initials (e.g. 'S', 'P', 'B')
    
    Examples:
    'Antony Selvan P.' -> 'antony selvan p'
    'Antony Selvan P'  -> 'antony selvan p'
    'ANTONY SELVAN P'  -> 'antony selvan p'
    'Antony   Selvan   P.' -> 'antony selvan p'
    'Antony Raj S'     -> 'antony raj s'
    'Antonyraj S'      -> 'antonyraj s'
    'Antony. S'        -> 'antony s'
    """
    if not name_str:
        return ""
    s = name_str.lower().strip()
    s = re.sub(r'[\.,\-_/\\\'"]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def determine_response_scope(query_text: str) -> str:
    """
    Determines the precise response scope from the user query:
    - 'BAPTISM_STATUS'
    - 'COMMUNION_STATUS'
    - 'CONFIRMATION_STATUS'
    - 'MARRIAGE_STATUS'
    - 'DEATH_STATUS'
    - 'ALL_SACRAMENTS'
    - 'MEMBER_PHONE'
    - 'MEMBER_ADDRESS'
    - 'FAMILY_MEMBERS_ONLY'
    - 'FAMILY_DETAILS'
    - 'GENERAL_MEMBER'
    """
    clean = re.sub(r'\s*\((?:Member\s*ID|Family\s*ID|Family|ID|Card)[:\s0-9A-Za-z,\s\-]+\)', '', query_text).strip()
    clean = re.sub(r'[\?!]+$', '', clean).strip()
    low = clean.lower()

    # 1. Phone / Mobile / Contact
    if bool(re.search(r'\b(?:phone|mobile|contact|cell)\b', low) or any(k in low for k in ['தொலைபேசி', 'அலைபேசி', 'தொடர்பு எண்', 'போன்'])):
        return 'MEMBER_PHONE'

    # 2. Address / Where live
    if bool(re.search(r'\b(?:address|residence)\b', low) or re.search(r'\bwhere\s+does\b.*\b(?:live|reside|stay)\b', low) or any(k in low for k in ['முகவரி', 'எங்கே வசிக்கிறார்', 'எங்கு வசிக்கிறார்'])):
        return 'MEMBER_ADDRESS'

    # 3. Family members only
    if bool(re.search(r'\b(?:family\s+members?|members\s+of\s+(?:the\s+)?family)\b', low) or re.search(r'\bwho\s+are\b.*\bmembers\b', low) or any(k in low for k in ['குடும்ப உறுப்பினர்கள்', 'குடும்ப அங்கத்தினர்கள்'])):
        return 'FAMILY_MEMBERS_ONLY'

    # 4. Family details
    if bool(re.search(r'\b(?:family\s+details?|family\s+info(?:rmation)?|family\s+records?|family\s+card|details\s+of\s+family)\b', low) or any(k in low for k in ['குடும்ப விவரம்', 'குடும்ப அட்டை'])):
        return 'FAMILY_DETAILS'

    # 5. Specific Sacraments (Individual status requested)
    is_all_sacs = bool(re.search(r'\b(?:all\s+sacraments?|all\s+sacramental)\b', low))

    if not is_all_sacs:
        if bool(re.search(r'\b(?:baptism|baptisms|baptised|baptized)\b', low) or any(k in low for k in ['ஞானஸ்நானம்', 'திருமுழுக்கு'])):
            return 'BAPTISM_STATUS'
        if bool(re.search(r'\b(?:communion|first\s+holy\s+communion|fhc|eucharist)\b', low) or any(k in low for k in ['நற்கருணை', 'முதல் நற்கருணை', 'புதுநன்மை'])):
            return 'COMMUNION_STATUS'
        if bool(re.search(r'\b(?:confirmation|confirmations|chrism)\b', low) or any(k in low for k in ['உறுதிப்பூசுதல்'])):
            return 'CONFIRMATION_STATUS'
        if bool(re.search(r'\b(?:marriage|marriages|matrimony|wedding|spouse|married)\b', low) or any(k in low for k in ['திருமணம்', 'விவாகம்'])):
            return 'MARRIAGE_STATUS'
        if bool(re.search(r'\b(?:death|deceased|burial|died)\b', low) or any(k in low for k in ['இறப்பு', 'அடக்கம்'])):
            return 'DEATH_STATUS'

    # 6. All Sacraments / Sacrament bundle
    if is_all_sacs or bool(re.search(r'\b(?:sacraments?\s+details?|sacraments?\s+records?|sacramental\s+status|sacraments?|sacrements?)\b', low) or any(k in low for k in ['அருட்சாதனம்', 'திருவருட்சாதனம்'])):
        return 'ALL_SACRAMENTS'

    if any(k in low for k in ['family of', 'household of']):
        return 'FAMILY_DETAILS'

    return 'GENERAL_MEMBER'


def build_candidate_prompt(query_text: str, person_name: str, cand_name: str, cand_mid: str) -> str:
    """
    Preserves the user's original query intent across disambiguation by substituting
    or appending the candidate's name and Member ID.
    """
    clean_q = re.sub(r'\s*\((?:Member\s*ID|Family\s*ID|Family|ID|Card)[:\s0-9A-Za-z,\s\-]+\)', '', query_text).strip()
    clean_q = re.sub(r'[\?!]+$', '', clean_q).strip()

    if person_name and person_name.lower() in clean_q.lower():
        pattern_poss = re.compile(re.escape(person_name) + r"('s|’s)", re.IGNORECASE)
        pattern = re.compile(re.escape(person_name), re.IGNORECASE)
        if pattern_poss.search(clean_q):
            replaced = pattern_poss.sub(f"{cand_name}'s", clean_q)
            return f"{replaced} (Member ID: {cand_mid})"
        else:
            return pattern.sub(f"{cand_name} (Member ID: {cand_mid})", clean_q)
    else:
        return f"{clean_q} for {cand_name} (Member ID: {cand_mid})"


def classify_query_intent(query_text: str) -> dict:
    """
    Classifies query intent and extracts person name & response scope:
    - 'COUNT_MEMBERS' / 'COUNT_FAMILIES'
    - 'LIST_MEMBERS' / 'LIST_FAMILIES'
    - Specific scopes: 'BAPTISM_STATUS', 'CONFIRMATION_STATUS', 'COMMUNION_STATUS',
      'MARRIAGE_STATUS', 'DEATH_STATUS', 'ALL_SACRAMENTS', 'MEMBER_PHONE',
      'MEMBER_ADDRESS', 'FAMILY_MEMBERS_ONLY', 'FAMILY_DETAILS', 'GENERAL_MEMBER'
    """
    q_clean = query_text.strip()
    q_low = q_clean.lower()

    # 1. COUNT INTENT
    count_patterns = [
        r'\b(?:how\s+many|total(?:\s+registered)?|number\s+of|count(?:\s+of)?)\s+(?:parish\s+)?(?:members|parishioners|people)\b',
        r'\b(?:how\s+many|total(?:\s+registered)?|number\s+of|count(?:\s+of)?)\s+(?:parish\s+)?families\b',
        r'^(?:total(?:\s+registered)?|number\s+of|count(?:\s+of)?)\s+(?:members|parishioners|families)$',
        r'^(?:total\s+registered\s+members)$'
    ]
    if any(re.search(p, q_low) for p in count_patterns) or q_low.strip() in ['total registered members', 'total members', 'member count']:
        if 'famil' in q_low:
            return {'intent': 'COUNT_FAMILIES', 'scope': 'COUNT_FAMILIES', 'person_name': None}
        return {'intent': 'COUNT_MEMBERS', 'scope': 'COUNT_MEMBERS', 'person_name': None}

    # 2. LIST INTENT
    starts_with_m = re.search(r'\b(?:name(?:s)?\s+(?:start(?:s)?|starting)\s+with|begins?\s+with)\s+([A-Za-z]+)\b', q_clean, re.IGNORECASE)
    name_filter = starts_with_m.group(1) if starts_with_m else None
    count_m = re.search(r'\b(\d+)\b', q_low)
    requested_count = int(count_m.group(1)) if count_m else 10

    list_patterns = [
        r'\b(?:list|show|give(?:\s+me)?|display|get)\s+(?:any\s+|some\s+)?(?:\d+\s+)?(?:parish\s+)?(?:members|parishioners)\b',
        r'\b(?:list|show|give(?:\s+me)?|display|get)\s+(?:any\s+|some\s+)?(?:\d+\s+)?(?:parish\s+)?families\b',
        r'\b(?:any|some|\d+)\s+(?:parish\s+)?(?:members|parishioners)\s+in\s+(?:my|the|this)?\s*parish\b',
        r'^(?:members|parishioners)\s+in\s+(?:my|the|this)?\s*parish$',
        r'^(?:list|show|give)\s+(?:any\s+|some\s+)?(?:members|parishioners)$'
    ]
    if any(re.search(p, q_low) for p in list_patterns) or ('whose names start with' in q_low and ('member' in q_low or 'parishioner' in q_low)):
        if 'famil' in q_low and 'member' not in q_low:
            return {'intent': 'LIST_FAMILIES', 'scope': 'LIST_FAMILIES', 'requested_count': requested_count, 'person_name': None}
        return {'intent': 'LIST_MEMBERS', 'scope': 'LIST_MEMBERS', 'requested_count': requested_count, 'name_filter': name_filter, 'person_name': None}

    # Strip explicit IDs for person extraction
    clean_no_id = re.sub(r'\s*\((?:Member\s*ID|Family\s*ID|Family|ID|Card)[:\s0-9A-Za-z,\s\-]+\)', '', q_clean).strip()
    clean_no_id = re.sub(r'[\?!]+$', '', clean_no_id).strip()

    # Determine Scope
    scope = determine_response_scope(clean_no_id)

    # Extract Person Name
    pname = None
    
    # Form A: Possessive '<prefix> <person>'s <attribute>'
    m_poss = re.search(r'(?:what\s+is|who\s+are|tell\s+me\s+about|give\s+me|show\s+me)?\s*(.+?)(?:\'s|’s)\s+(?:mobile|phone|contact|address|family|baptism|confirmation|communion|marriage|death|sacrament|details|records?|status|info)', clean_no_id, re.IGNORECASE)
    if m_poss:
        pname = m_poss.group(1).strip()
        pname = re.sub(r'^(?:what\s+is|who\s+are|tell\s+me\s+about|give\s+me|show\s+me|the|a|an)\s+', '', pname, flags=re.IGNORECASE).strip()

    # Form B: 'where does <person> live/reside/stay'
    if not pname:
        m_live = re.search(r'\bwhere\s+does\s+(.+?)\s+(?:live|reside|stay)\b', clean_no_id, re.IGNORECASE)
        if m_live:
            pname = m_live.group(1).strip()

    # Form C: '<query> (of|for|about) <person>'
    if not pname:
        m_of = re.search(r'\b(?:of|for|about)\s+([A-Za-z0-9\.\s]+)$', clean_no_id, re.IGNORECASE)
        if m_of:
            pname = m_of.group(1).strip()

    # Form D: Direct search 'show/find/view/get/who is <person>'
    if not pname:
        m_direct = re.search(r'^(?:show|find|view|get|who\s+is|tell\s+me\s+about)\s+(.+)$', clean_no_id, re.IGNORECASE)
        if m_direct:
            pname = m_direct.group(1).strip()

    # Form E: Short direct query
    if not pname and len(clean_no_id.split()) <= 4 and not any(w in clean_no_id.lower() for w in ['how', 'what', 'why', 'when', 'where', 'list', 'show']):
        pname = clean_no_id

    # Validation & stripping of any leftover generic/intent words
    if pname:
        pname = re.sub(r'^(?:the|a|an|parishioner|member)\s+', '', pname, flags=re.IGNORECASE).strip()
        raw_words = pname.split()
        filtered_words = [
            w for w in raw_words
            if w.lower().strip('.,?!\'":;') not in RESERVED_GENERIC_WORDS
            and not w.strip('.,?!\'":;').isdigit()
        ]
        if not filtered_words:
            pname = None
        else:
            pname = " ".join(filtered_words)

    intent = scope if pname else ('GENERAL_QUERY' if scope == 'GENERAL_MEMBER' else scope)
    return {
        'intent': intent,
        'scope': scope,
        'person_name': pname
    }

def extract_intent_and_person(query_text: str) -> tuple[str, str]:
    """
    Separates query intent from person name.
    Uses classify_query_intent to guarantee generic words are NEVER returned as person names.
    """
    res = classify_query_intent(query_text)
    intent = res.get('intent', 'GENERAL_QUERY').lower()
    pname = res.get('person_name') or ''
    return intent, pname

def extract_person_name_from_query(query_text: str) -> str:
    """Extracts only the verified person name from query text."""
    _, person_name = extract_intent_and_person(query_text)
    return person_name

def detect_query_intent(query_text: str) -> str:
    """Extracts only the intent from query text."""
    intent, _ = extract_intent_and_person(query_text)
    return intent

def handle_list_members(
    requested_count: int = 10,
    name_filter: str = None,
    user_parish: str = None,
    user_diocese: str = None,
    user_vicariate: str = None
) -> dict:
    """
    Directly queries and returns authorized parish members up to requested_count.
    Does NOT invoke person-name matching or candidate selection UI.
    """
    requested_count = max(1, min(requested_count, 100))
    where_clauses = []
    params = []
    
    if user_parish:
        where_clauses.append("(m.parish_id = %s OR m.parish_id LIKE %s)")
        params.extend([user_parish, f"%{user_parish}%"])
    elif user_vicariate:
        where_clauses.append("m.vicariate_id = %s")
        params.append(user_vicariate)
    elif user_diocese and user_diocese != "All Dioceses":
        where_clauses.append("m.diocese_id = %s")
        params.append(user_diocese)

    if name_filter:
        where_clauses.append("(m.first_name LIKE %s OR CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) LIKE %s)")
        params.extend([f"{name_filter}%", f"{name_filter}%"])

    base_where = " AND ".join(where_clauses) if where_clauses else "1=1"

    sql = f"""
        SELECT 
            m.name as member_id,
            TRIM(CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name)) as full_name,
            m.gender,
            m.mobile,
            m.parish_id,
            f.name as family_id,
            f.family_register_number as family_card,
            f.reference as family_name
        FROM `tabMember` m
        LEFT JOIN `tabFamily` f ON f.name = m.family_id
        WHERE {base_where}
        ORDER BY m.name ASC
        LIMIT %s
    """
    
    try:
        members = frappe.db.sql(sql, tuple(params + [requested_count]), as_dict=True)
    except Exception as e:
        return {
            "reply": f"Error retrieving parish members: {str(e)}",
            "generated_sql": sql,
            "data": [],
            "disambiguation": None,
            "suggested_questions": [],
            "query_id": -1
        }

    parish_label = user_parish or user_diocese or "your authorized jurisdiction"
    filter_note = f" whose names start with '{name_filter}'" if name_filter else ""
    
    if not members:
        reply = f"No registered members found in {parish_label}{filter_note}."
        return {
            "reply": reply,
            "generated_sql": sql,
            "data": [],
            "disambiguation": None,
            "suggested_questions": [],
            "query_id": -1
        }

    actual_count = len(members)
    header_count_str = f"{actual_count} members" if actual_count == requested_count else f"{actual_count} member(s) (all available records)"
    
    lines = [f"Here are {header_count_str} from your authorized parish ({parish_label}){filter_note}:\n"]
    lines.append("| # | Member Name | Family Card | Family Name | Gender | Contact |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    ui_members = []
    for idx, m in enumerate(members, 1):
        m_name = m.get("full_name") or "Member"
        f_card = m.get("family_card") or "-"
        f_name = m.get("family_name") or "-"
        gender = m.get("gender") or "-"
        phone = m.get("mobile") or "-"
        lines.append(f"| {idx} | **{m_name}** | `{f_card}` | {f_name} | {gender} | {phone} |")
        ui_members.append({
            "#": idx,
            "Member Name": m_name,
            "Family Card": f_card,
            "Family Name": f_name,
            "Gender": gender,
            "Contact": phone,
        })

    suggestions = []
    if members:
        top_m = members[0].get("full_name")
        suggestions = [
            f"Family details of {top_m}",
            f"Sacrament details of {top_m}",
            "How many members are in my parish"
        ]

    return {
        "reply": "\n".join(lines),
        "generated_sql": sql,
        "data": ui_members,
        "disambiguation": None,
        "suggested_questions": suggestions,
        "query_id": -1
    }

def handle_count_members(
    user_parish: str = None,
    user_diocese: str = None,
    user_vicariate: str = None
) -> dict:
    """
    Directly queries and returns total count of authorized parish members.
    """
    where_clauses = []
    params = []
    
    if user_parish:
        where_clauses.append("(m.parish_id = %s OR m.parish_id LIKE %s)")
        params.extend([user_parish, f"%{user_parish}%"])
    elif user_vicariate:
        where_clauses.append("m.vicariate_id = %s")
        params.append(user_vicariate)
    elif user_diocese and user_diocese != "All Dioceses":
        where_clauses.append("m.diocese_id = %s")
        params.append(user_diocese)

    base_where = " AND ".join(where_clauses) if where_clauses else "1=1"
    sql = f"SELECT COUNT(*) FROM `tabMember` m WHERE {base_where}"

    try:
        cnt = frappe.db.sql(sql, tuple(params))[0][0]
    except Exception as e:
        return {
            "reply": f"Error counting registered members: {str(e)}",
            "generated_sql": sql,
            "data": [],
            "disambiguation": None,
            "suggested_questions": [],
            "query_id": -1
        }

    parish_label = user_parish or user_diocese or "your authorized jurisdiction"
    reply = f"There are currently **{cnt}** registered members in {parish_label}."
    
    return {
        "reply": reply,
        "generated_sql": sql,
        "data": [{"total_members": cnt, "jurisdiction": parish_label}],
        "disambiguation": None,
        "suggested_questions": [
            "List any 10 members in my parish",
            "How many families are in my parish",
            "Total registered families"
        ],
        "query_id": -1
    }

def handle_list_families(
    requested_count: int = 10,
    user_parish: str = None,
    user_diocese: str = None,
    user_vicariate: str = None
) -> dict:
    """
    Directly queries and returns authorized families up to requested_count.
    """
    requested_count = max(1, min(requested_count, 100))
    where_clauses = []
    params = []
    
    if user_parish:
        where_clauses.append("(f.parish_id = %s OR f.parish_id LIKE %s)")
        params.extend([user_parish, f"%{user_parish}%"])
    elif user_vicariate:
        where_clauses.append("f.vicariate_id = %s")
        params.append(user_vicariate)
    elif user_diocese and user_diocese != "All Dioceses":
        where_clauses.append("f.diocese_id = %s")
        params.append(user_diocese)

    base_where = " AND ".join(where_clauses) if where_clauses else "1=1"
    sql = f"""
        SELECT 
            f.name as family_id,
            f.family_register_number as family_card,
            f.reference as family_name,
            f.parish_bcc_id as anbiyam,
            f.street,
            f.mobile,
            f.parish_id
        FROM `tabFamily` f
        WHERE {base_where}
        ORDER BY f.name ASC
        LIMIT %s
    """
    
    try:
        families = frappe.db.sql(sql, tuple(params + [requested_count]), as_dict=True)
    except Exception as e:
        return {
            "reply": f"Error retrieving families: {str(e)}",
            "generated_sql": sql,
            "data": [],
            "disambiguation": None,
            "suggested_questions": [],
            "query_id": -1
        }

    parish_label = user_parish or user_diocese or "your authorized jurisdiction"
    if not families:
        return {
            "reply": f"No registered families found in {parish_label}.",
            "generated_sql": sql,
            "data": [],
            "disambiguation": None,
            "suggested_questions": [],
            "query_id": -1
        }

    actual_count = len(families)
    header_count_str = f"{actual_count} families" if actual_count == requested_count else f"{actual_count} family/families (all available records)"
    
    lines = [f"Here are {header_count_str} from your authorized parish ({parish_label}):\n"]
    lines.append("| # | Family Name | Family Card | BCC / Anbiyam | Address | Contact |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    ui_families = []
    for idx, f in enumerate(families, 1):
        f_name = f.get("family_name") or "Family"
        f_card = f.get("family_card") or "-"
        anbiyam = f.get("anbiyam") or "-"
        street = f.get("street") or "-"
        mob = f.get("mobile") or "-"
        lines.append(f"| {idx} | **{f_name}** | `{f_card}` | {anbiyam} | {street} | {mob} |")
        ui_families.append({
            "#": idx,
            "Family Name": f_name,
            "Family Card": f_card,
            "BCC / Anbiyam": anbiyam,
            "Address": street,
            "Contact": mob,
        })

    return {
        "reply": "\n".join(lines),
        "generated_sql": sql,
        "data": ui_families,
        "disambiguation": None,
        "suggested_questions": [f"Family details of {families[0].get('family_name')}"],
        "query_id": -1
    }

def handle_count_families(
    user_parish: str = None,
    user_diocese: str = None,
    user_vicariate: str = None
) -> dict:
    """
    Directly queries and returns total count of authorized registered families.
    """
    where_clauses = []
    params = []
    
    if user_parish:
        where_clauses.append("(f.parish_id = %s OR f.parish_id LIKE %s)")
        params.extend([user_parish, f"%{user_parish}%"])
    elif user_vicariate:
        where_clauses.append("f.vicariate_id = %s")
        params.append(user_vicariate)
    elif user_diocese and user_diocese != "All Dioceses":
        where_clauses.append("f.diocese_id = %s")
        params.append(user_diocese)

    base_where = " AND ".join(where_clauses) if where_clauses else "1=1"
    sql = f"SELECT COUNT(*) FROM `tabFamily` f WHERE {base_where}"

    try:
        cnt = frappe.db.sql(sql, tuple(params))[0][0]
    except Exception as e:
        return {
            "reply": f"Error counting registered families: {str(e)}",
            "generated_sql": sql,
            "data": [],
            "disambiguation": None,
            "suggested_questions": [],
            "query_id": -1
        }

    parish_label = user_parish or user_diocese or "your authorized jurisdiction"
    reply = f"There are currently **{cnt}** registered families in {parish_label}."
    
    return {
        "reply": reply,
        "generated_sql": sql,
        "data": [{"total_families": cnt, "jurisdiction": parish_label}],
        "disambiguation": None,
        "suggested_questions": [
            "List any 10 families in my parish",
            "List any 10 members in my parish",
            "Total registered members"
        ],
        "query_id": -1
    }

PHONETIC_REPLACEMENTS = [
    (r'\bantoney\b', 'antony'),
    (r'\banthoney\b', 'antony'),
    (r'\banthony\b', 'antony'),
    (r'\bantoni\b', 'antony'),
    (r'\bantosy\b', 'antony'),
    (r'\bselvam\b', 'selvan'),
    (r'\barockiaraj\b', 'arokiaraj'),
    (r'\barockiya\s+raj\b', 'arokiaraj'),
    (r'\barokiya\s+raj\b', 'arokiaraj'),
    (r'\barokiyaraj\b', 'arokiaraj'),
    (r'\bmariya\b', 'maria'),
    (r'\bsebastiyan\b', 'sebastian'),
    (r'\biruthayaraj\b', 'irudayaraj'),
]

def split_base_and_initials(norm_name: str) -> tuple[list[str], str, list[str]]:
    tokens = norm_name.split()
    base_tokens = [t for t in tokens if len(t) > 1]
    initials = [t for t in tokens if len(t) == 1]
    if not base_tokens and tokens:
        base_tokens = tokens
        initials = []
    base_name = " ".join(base_tokens)
    return base_tokens, base_name, initials

def phonetic_normalize(text: str) -> str:
    s = text
    for pat, repl in PHONETIC_REPLACEMENTS:
        s = re.sub(pat, repl, s)
    tokens = []
    for t in s.split():
        if len(t) >= 5 and t.endswith('am'):
            tokens.append(t[:-1] + 'n')
        else:
            tokens.append(t)
    return " ".join(tokens)

def compute_member_similarity(norm_query: str, norm_full: str) -> tuple[float, str]:
    """
    Computes complete-name similarity between norm_query and norm_full.
    Accounts for base name vs optional single-letter initials, phonetic variations,
    and penalizes partial single-word hijacking.
    Returns (score_0_to_100, match_category).
    """
    if not norm_query or not norm_full:
        return 0.0, "NONE"

    # 1. Exact normalized full name match
    if norm_query == norm_full:
        return 100.0, "EXACT_FULL"

    q_tokens, q_base, q_inits = split_base_and_initials(norm_query)
    m_tokens, m_base, m_inits = split_base_and_initials(norm_full)

    # Check if user explicitly supplied initials that conflict with member's initials
    init_penalty = 0.0
    if q_inits:
        if m_inits and not set(q_inits).intersection(set(m_inits)):
            init_penalty = 12.0
        elif not m_inits:
            init_penalty = 5.0

    # 2. Single-token query handling (e.g., "Antony", "Selvam", or with initial "Antony S")
    if len(q_tokens) == 1:
        q_tok = q_tokens[0]
        q_phon = phonetic_normalize(q_tok)
        m_phon_tokens = [phonetic_normalize(t) for t in m_tokens]

        # Compound single-token query matching multi-token base (e.g. "antonyraj" vs "antony raj")
        spaceless_sim = max(
            fuzz.ratio(q_tok, m_base.replace(" ", "")),
            fuzz.ratio(q_phon, phonetic_normalize(m_base.replace(" ", "")))
        )
        if len(m_tokens) >= 2 and spaceless_sim >= 92.0:
            score = spaceless_sim - init_penalty
            return round(score, 1), "COMPOUND_BASE"

        # If the user ALSO provided a matching initial (e.g. "Antony S" or "Selvam A")
        if len(m_tokens) == 1 and q_inits and set(q_inits) == set(m_inits):
            raw = max(fuzz.ratio(q_tok, m_tokens[0]), fuzz.ratio(q_phon, m_phon_tokens[0]) * 0.96)
            return round(raw, 1), ("EXACT_BASE" if raw >= 95.0 else "PHONETIC_BASE")

        # Un-initialed single-word query (e.g. "Antony")
        first_exact = fuzz.ratio(q_tok, m_tokens[0])
        first_phon = fuzz.ratio(q_phon, m_phon_tokens[0])
        first_tok_sim = max(first_exact, first_phon * 0.98)

        if first_tok_sim >= 88.0:
            # All members whose first given name is "Antony" (e.g. Antony Selvan P, Antony Raj S, Anthony S)
            # receive comparable prefix scores (~81.5-83.0) so multiple "Antony ..." members trigger SHOW_CANDIDATES,
            # while a truly unique single given name in the parish still wins by score_gap >= 15.0.
            exact_bonus = 1.5 if first_exact == 100.0 else 0.0
            score = 81.5 + exact_bonus - init_penalty
            return round(score, 1), "PREFIX_TOKEN"

        best_tok_sim = max(
            max(fuzz.ratio(q_tok, mt), fuzz.ratio(q_phon, mpt) * 0.98)
            for mt, mpt in zip(m_tokens, m_phon_tokens)
        )
        if best_tok_sim >= 85.0:
            score = 70.0 + (best_tok_sim - 85.0) * 0.2 - init_penalty
            return round(min(score, 74.0), 1), "PARTIAL_TOKEN"
        else:
            return round(fuzz.ratio(q_base, m_base) * 0.8, 1), "LOW"

    # 3. Multi-token query handling (e.g. "Antony Selvam", "Antony Raj")
    q_phon_base = phonetic_normalize(q_base)
    m_phon_base = phonetic_normalize(m_base)

    # Exact base match (ignoring optional trailing initial in DB)
    if q_base == m_base and init_penalty == 0.0:
        return 98.5, "EXACT_BASE"

    # Exact phonetic base match (e.g., "antony selvam" == "antony selvan")
    if q_phon_base == m_phon_base and init_penalty == 0.0:
        return 96.0, "PHONETIC_BASE"

    # Compound / spaceless base match (e.g. "antony raj" vs "antonyraj")
    q_spaceless = q_base.replace(" ", "")
    m_spaceless = m_base.replace(" ", "")
    if q_spaceless == m_spaceless:
        return round(95.0 - init_penalty, 1), "COMPOUND_BASE"
    if phonetic_normalize(q_spaceless) == phonetic_normalize(m_spaceless):
        return round(93.5 - init_penalty, 1), "PHONETIC_COMPOUND"

    # General multi-token comparison
    len_ratio = min(len(q_spaceless), len(m_spaceless)) / max(len(q_spaceless), len(m_spaceless))
    base_ratio = fuzz.ratio(q_base, m_base)
    phon_ratio = fuzz.ratio(q_phon_base, m_phon_base)
    spaceless_ratio = fuzz.ratio(q_spaceless, m_spaceless)

    raw_sim = max(base_ratio, phon_ratio * 0.98, spaceless_ratio * 0.97)

    # Penalize heavily if member is missing a major token (e.g. "Selvam A" vs "Antony Selvam")
    if len(m_tokens) < len(q_tokens) and len_ratio < 0.75:
        raw_sim = raw_sim * (0.55 + 0.35 * len_ratio)

    final_score = max(0.0, raw_sim - init_penalty)
    return round(final_score, 1), "FUZZY"


def resolve_member_and_family(
    query_text: str,
    user_parish: str = None,
    user_diocese: str = None,
    user_vicariate: str = None
) -> dict:
    """
    High-Precision 3-Level Member and Family Resolver:
    LEVEL 1 — EXACT MATCH (Full name or exact base name)
    LEVEL 2 — UNIQUE HIGH-CONFIDENCE MATCH (Score threshold + Score Gap rule)
    LEVEL 3 — FUZZY AMBIGUOUS MATCH (Multiple close high-scoring candidates)
    """
    if not query_text or not query_text.strip():
        return {"status": "empty"}

    intent_res = classify_query_intent(query_text)
    intent = intent_res.get("intent", "GENERAL_QUERY").lower()
    person_name = intent_res.get("person_name") or ""
    scope = intent_res.get("scope", "GENERAL_MEMBER")

    if not person_name:
        return {"status": "not_found", "intent": intent, "response_scope": scope}

    # Reject if person_name is only generic words
    tokens = [w.lower().strip('.') for w in person_name.split()]
    if not tokens or all(w in RESERVED_GENERIC_WORDS or w.isdigit() for w in tokens):
        return {"status": "not_found", "intent": intent, "response_scope": scope}

    norm_query = normalize_name(person_name)
    if len(norm_query) < 2:
        return {"status": "not_found", "intent": intent, "response_scope": scope}

    # 1. Build Pre-Search Jurisdiction Filter (Permission filtering BEFORE matching)
    where_clauses = []
    params = []
    
    if user_parish:
        where_clauses.append("(m.parish_id = %s OR m.parish_id LIKE %s)")
        params.extend([user_parish, f"%{user_parish}%"])
    elif user_vicariate:
        where_clauses.append("m.vicariate_id = %s")
        params.append(user_vicariate)
    elif user_diocese and user_diocese != "All Dioceses":
        where_clauses.append("m.diocese_id = %s")
        params.append(user_diocese)

    base_where = " AND ".join(where_clauses) if where_clauses else "1=1"

    sql = f"""
        SELECT 
            m.name as member_id,
            m.first_name,
            m.middle_name,
            m.last_name,
            TRIM(CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name)) as full_name,
            m.family_id,
            m.parish_id,
            m.gender,
            m.mobile,
            m.dob,
            m.age,
            m.bapt_date,
            m.fhc_date,
            m.cnf_date,
            m.mrg_date,
            f.family_register_number,
            f.reference as family_name,
            f.parish_bcc_id as anbiyam,
            f.street as family_address
        FROM `tabMember` m
        LEFT JOIN `tabFamily` f ON f.name = m.family_id
        WHERE {base_where}
    """
    
    try:
        members = frappe.db.sql(sql, tuple(params), as_dict=True)
    except Exception as e:
        print(f"[name_search] Error fetching members: {e}")
        return {"status": "error", "error": str(e), "intent": intent, "response_scope": scope}

    if not members:
        return {"status": "not_found", "intent": intent, "response_scope": scope}

    # Deduplicate members by member_id and score all authorized members
    seen_member_ids = set()
    all_scored = []
    exact_matches = []

    for m in members:
        if m.member_id in seen_member_ids:
            continue
        seen_member_ids.add(m.member_id)
        norm_full = normalize_name(m.full_name)
        if norm_full == norm_query:
            exact_matches.append(m)
        score, category = compute_member_similarity(norm_query, norm_full)
        all_scored.append((score, category, m))

    all_scored.sort(key=lambda x: x[0], reverse=True)

    top_score, top_cat, top_m = all_scored[0] if all_scored else (0.0, "NONE", None)
    second_score, second_cat, second_m = all_scored[1] if len(all_scored) > 1 else (0.0, "NONE", None)
    score_gap = round(top_score - second_score, 1)

    # 3-LEVEL DECISION LOGIC
    exact_match_count = len(exact_matches)
    if exact_match_count == 1:
        decision = "EXACT_MATCH"
        action = "DIRECT_RESULT"
        selected_member = exact_matches[0]
    elif exact_match_count > 1:
        decision = "MULTIPLE_EXACT_MATCHES"
        action = "SHOW_CANDIDATES"
        selected_member = None
    elif top_score >= 85.0 and (second_score < 75.0 or score_gap >= 8.0):
        # LEVEL 2 — UNIQUE HIGH-CONFIDENCE MATCH (e.g. "Antony Selvam" -> "Antony Selvan P" 96.0 vs 69.6)
        decision = "UNIQUE_HIGH_CONFIDENCE_MATCH"
        action = "DIRECT_RESULT"
        selected_member = top_m
    elif top_score >= 80.0 and score_gap >= 15.0:
        # LEVEL 2 — UNIQUE HIGH-CONFIDENCE MATCH with wide margin
        decision = "UNIQUE_HIGH_CONFIDENCE_MATCH"
        action = "DIRECT_RESULT"
        selected_member = top_m
    elif top_score >= 74.0:
        # Check how many candidates are genuinely competitive with top_score (within 12 points and >= 74.0)
        competitive = [item for item in all_scored if item[0] >= 74.0 and (top_score - item[0]) <= 12.0]
        if len(competitive) == 1 and top_score >= 80.0:
            decision = "UNIQUE_HIGH_CONFIDENCE_MATCH"
            action = "DIRECT_RESULT"
            selected_member = top_m
        else:
            decision = "MULTIPLE_CLOSE_MATCHES"
            action = "SHOW_CANDIDATES"
            selected_member = None
    else:
        decision = "NO_MATCH"
        action = "NO_MATCH"
        selected_member = None

    # Mandatory Debug Output (Section 18)
    print("=" * 60)
    print(f"ORIGINAL QUERY: {query_text}")
    print(f"EXTRACTED PERSON NAME: {person_name}")
    print(f"NORMALIZED QUERY: {norm_query}")
    print(f"TOP MATCH: {top_m.full_name if top_m else 'None'} (ID: {top_m.member_id if top_m else 'None'})")
    print(f"TOP SCORE: {top_score} [{top_cat}]")
    print(f"SECOND MATCH: {second_m.full_name if second_m else 'None'} (ID: {second_m.member_id if second_m else 'None'})")
    print(f"SECOND SCORE: {second_score} [{second_cat}]")
    print(f"SCORE GAP: {score_gap}")
    print(f"MATCH DECISION: {decision}")
    print(f"ACTION: {action}")
    print("=" * 60)

    # EXECUTE ACTION
    if action == "DIRECT_RESULT" and selected_member:
        fam_bundle = fetch_full_family_bundle(selected_member.family_id, selected_member.parish_id)
        sac_bundle = fetch_member_sacrament_bundle(selected_member.member_id, selected_member.parish_id)
        return {
            "status": "exact",
            "matched_member": selected_member,
            "family_bundle": fam_bundle,
            "sacrament_bundle": sac_bundle,
            "intent": intent,
            "response_scope": scope,
            "member_id": selected_member.member_id,
            "family_id": selected_member.family_id,
            "match_decision": decision,
            "top_score": top_score
        }

    if action == "SHOW_CANDIDATES":
        if exact_match_count > 1:
            cand_Pool = [(100.0, "EXACT_FULL", m) for m in exact_matches[:3]]
        else:
            cand_Pool = [item for item in all_scored if item[0] >= 74.0 and (top_score - item[0]) <= 12.0][:3]

        top_candidates = []
        for score, cat, m in cand_Pool:
            cand_prompt = build_candidate_prompt(query_text, person_name, m.full_name, m.member_id)
            top_candidates.append({
                "type": "member",
                "member_id": m.member_id,
                "full_name": m.full_name,
                "family_id": m.family_id,
                "family_card": m.family_register_number or m.family_id,
                "card_no": m.family_register_number or m.family_id,
                "family_name": m.family_name or "Family",
                "anbiyam": m.anbiyam or "",
                "place": m.family_address or m.parish_id,
                "parish_id": m.parish_id,
                "prompt": cand_prompt,
                "display_text": f"{m.full_name} ({m.family_name or m.family_register_number})",
                "similarity": round(score, 1)
            })

        return {
            "status": "candidates",
            "candidates": top_candidates,
            "intent": intent,
            "response_scope": scope,
            "match_decision": decision
        }

    return {"status": "not_found", "intent": intent, "response_scope": scope, "match_decision": decision}

def fetch_full_family_bundle(family_id: str, parish_id: str = None) -> dict:
    """
    Authoritative Family & Member Retrieval:
    Uses member.family_id -> tabFamily.name = family_id
    Then retrieves all members where family_id = family_id
    """
    if not family_id:
        return {}

    fam = frappe.db.get_value(
        "Family",
        family_id,
        [
            "name", "family_register_number", "reference", "parish_bcc_id",
            "zone_id", "lang_community_id", "rite_id", "status", "active",
            "street", "city", "zip", "phone", "mobile", "email",
            "house_ownership", "is_civil_marriage", "is_church_marriage",
            "register_date", "parish_id"
        ],
        as_dict=True
    ) or {}

    members = frappe.db.sql(
        """
        SELECT 
            name as member_id,
            first_name, middle_name, last_name,
            TRIM(CONCAT_WS(' ', first_name, middle_name, last_name)) as full_name,
            gender, mobile, email, dob, age, relationship_id, is_family_head,
            bapt_date, fhc_date, cnf_date, mrg_date
        FROM `tabMember`
        WHERE family_id = %s
        ORDER BY FIELD(relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, name ASC
        """,
        (family_id,),
        as_dict=True
    ) or []

    card_no = fam.get("family_register_number") or family_id
    parish = parish_id or fam.get("parish_id")

    sacraments = {
        "baptisms": [],
        "communions": [],
        "confirmations": [],
        "marriages": [],
        "deaths": []
    }

    try:
        sacraments["baptisms"] = frappe.db.sql(
            "SELECT name, first_name, last_name, bapt_date, bapt_place, bapt_minister, bapt_god_father, bapt_god_mother FROM `tabBaptism` WHERE (family_card_no = %s OR family_card_no = %s) AND parish_id = %s",
            (card_no, family_id, parish),
            as_dict=True
        ) or []

        sacraments["communions"] = frappe.db.sql(
            "SELECT name, first_name, last_name, fhc_date, fhc_place, fhc_minister FROM `tabCommunion` WHERE (family_card_no = %s OR family_card_no = %s) AND parish_id = %s",
            (card_no, family_id, parish),
            as_dict=True
        ) or []

        sacraments["confirmations"] = frappe.db.sql(
            "SELECT name, first_name, last_name, cnf_date, cnf_place, cnf_minister FROM `tabConfirmation` WHERE (family_card_no = %s OR family_card_no = %s) AND parish_id = %s",
            (card_no, family_id, parish),
            as_dict=True
        ) or []

        sacraments["marriages"] = frappe.db.sql(
            "SELECT name, bridegroom_name, bride_name, mrg_date, mrg_place, mrg_minister FROM `tabMarriage` WHERE (family_card_no = %s OR family_card_no = %s) AND parish_id = %s",
            (card_no, family_id, parish),
            as_dict=True
        ) or []

        sacraments["deaths"] = frappe.db.sql(
            "SELECT name, first_name, last_name, death_date, burial_date, burial_place FROM `tabDeath` WHERE (family_card_no = %s OR family_card_no = %s) AND parish_id = %s",
            (card_no, family_id, parish),
            as_dict=True
        ) or []
    except Exception as se:
        print(f"[name_search] Error fetching family sacraments: {se}")

    return {
        "family": fam,
        "members": members,
        "sacraments": sacraments
    }

def fetch_member_sacrament_bundle(member_id: str, parish_id: str = None) -> dict:
    """
    Retrieves sacrament register records specifically for a single member.
    """
    if not member_id:
        return {}

    mem = frappe.db.get_value(
        "Member",
        member_id,
        ["name", "first_name", "middle_name", "last_name", "family_id", "parish_id", "bapt_date", "fhc_date", "cnf_date", "mrg_date"],
        as_dict=True
    ) or {}

    full_name = f"{mem.get('first_name') or ''} {mem.get('middle_name') or ''} {mem.get('last_name') or ''}".strip()
    full_name = re.sub(r'\s+', ' ', full_name)

    bap = None
    fhc = None
    cnf = None
    mrg = None
    dth = None

    try:
        baps = frappe.db.sql(
            "SELECT name, first_name, last_name, bapt_date, bapt_place, bapt_minister, bapt_god_father, bapt_god_mother FROM `tabBaptism` WHERE member_id = %s OR (first_name = %s AND (family_card_no = %s OR last_name = %s)) LIMIT 1",
            (member_id, mem.get("first_name"), mem.get("family_id"), mem.get("last_name")),
            as_dict=True
        )
        if baps:
            bap = baps[0]
    except Exception:
        pass

    try:
        fhcs = frappe.db.sql(
            "SELECT name, first_name, last_name, fhc_date, fhc_place, fhc_minister FROM `tabCommunion` WHERE member_id = %s OR (first_name = %s AND family_card_no = %s) LIMIT 1",
            (member_id, mem.get("first_name"), mem.get("family_id")),
            as_dict=True
        )
        if fhcs:
            fhc = fhcs[0]
    except Exception:
        pass

    try:
        cnfs = frappe.db.sql(
            "SELECT name, first_name, last_name, cnf_date, cnf_place, cnf_minister FROM `tabConfirmation` WHERE member_id = %s OR (first_name = %s AND family_card_no = %s) LIMIT 1",
            (member_id, mem.get("first_name"), mem.get("family_id")),
            as_dict=True
        )
        if cnfs:
            cnf = cnfs[0]
    except Exception:
        pass

    try:
        mrgs = frappe.db.sql(
            "SELECT name, bridegroom_name, bride_name, mrg_date, mrg_place, mrg_minister FROM `tabMarriage` WHERE (bridegroom_id = %s OR bride_id = %s) OR (bridegroom_name LIKE %s OR bride_name LIKE %s) LIMIT 1",
            (member_id, member_id, f"%{mem.get('first_name')}%", f"%{mem.get('first_name')}%"),
            as_dict=True
        )
        if mrgs:
            mrg = mrgs[0]
    except Exception:
        pass

    return {
        "member": mem,
        "full_name": full_name,
        "baptism": bap,
        "communion": fhc,
        "confirmation": cnf,
        "marriage": mrg,
        "death": dth
    }

def render_scoped_response(
    scope: str,
    target_member: dict,
    sac_bundle: dict = None,
    fam_bundle: dict = None
) -> tuple[str, list[str]]:
    """
    Renders ONLY what the user asked based on scope:
    - BAPTISM_STATUS: Only Baptism status sentence/details.
    - COMMUNION_STATUS: Only First Holy Communion status sentence/details.
    - CONFIRMATION_STATUS: Only Confirmation status sentence/details.
    - MARRIAGE_STATUS: Only Marriage status sentence/details.
    - DEATH_STATUS: Only Death status sentence/details.
    - ALL_SACRAMENTS: All sacraments register overview (no family table).
    - MEMBER_PHONE: ONLY mobile/phone number.
    - MEMBER_ADDRESS: ONLY address.
    - FAMILY_MEMBERS_ONLY: ONLY family members table (no family metadata dump, no sacraments).
    - FAMILY_DETAILS: Family metadata + family members table (no sacraments).
    - GENERAL_MEMBER: Member overview card.
    
    Returns (reply_markdown, dynamic_suggested_questions).
    """
    full_name = target_member.get("full_name") or target_member.get("first_name") or "Parishioner"
    parish = target_member.get("parish_id") or "the parish"
    fam = (fam_bundle.get("family") if fam_bundle else None) or {}
    card_no = fam.get("family_register_number") or target_member.get("family_card") or target_member.get("family_id") or "N/A"

    if scope == 'BAPTISM_STATUS':
        bap_rec = sac_bundle.get("baptism") if sac_bundle else None
        bapt_date = (bap_rec.get("bapt_date") if bap_rec else None) or target_member.get("bapt_date")
        if bapt_date or bap_rec:
            d_str = str(bapt_date) if bapt_date else "Recorded in parish register"
            place = (bap_rec.get("bapt_place") if bap_rec else None) or parish
            minister = bap_rec.get("bapt_minister") if bap_rec else None
            godfather = bap_rec.get("bapt_god_father") if bap_rec else None
            
            lines = [
                f"**{full_name}** has received the Sacrament of Baptism.",
                f"- **Date:** {d_str}",
                f"- **Parish / Place:** {place}"
            ]
            if minister:
                lines.append(f"- **Minister:** {minister}")
            if godfather:
                lines.append(f"- **Godparent:** {godfather}")
            reply = "\n".join(lines)
        else:
            reply = f"**{full_name}** has no baptism record in {parish}."
        
        suggestions = [
            f"What is {full_name}'s confirmation status?",
            f"What is {full_name}'s First Holy Communion status?",
            f"Show all sacrament details of {full_name}"
        ]

    elif scope == 'COMMUNION_STATUS':
        fhc_rec = sac_bundle.get("communion") if sac_bundle else None
        fhc_date = (fhc_rec.get("fhc_date") if fhc_rec else None) or target_member.get("fhc_date")
        if fhc_date or fhc_rec:
            d_str = str(fhc_date) if fhc_date else "Recorded in parish register"
            reply = f"**{full_name}** has received the Sacrament of First Holy Communion.\n- **Date:** {d_str}"
        else:
            reply = f"**{full_name}** has no First Holy Communion record in {parish}."
        suggestions = [
            f"What is {full_name}'s baptism status?",
            f"What is {full_name}'s confirmation status?",
            f"Show all sacrament details of {full_name}"
        ]

    elif scope == 'CONFIRMATION_STATUS':
        cnf_rec = sac_bundle.get("confirmation") if sac_bundle else None
        cnf_date = (cnf_rec.get("cnf_date") if cnf_rec else None) or target_member.get("cnf_date")
        if cnf_date or cnf_rec:
            d_str = str(cnf_date) if cnf_date else "Recorded in parish register"
            place = (cnf_rec.get("cnf_place") if cnf_rec else None) or parish
            reply = f"**{full_name}** has received the Sacrament of Confirmation.\n- **Date:** {d_str}\n- **Parish / Place:** {place}"
        else:
            reply = f"**{full_name}** has no confirmation record in {parish}."
        suggestions = [
            f"What is {full_name}'s baptism status?",
            f"What is {full_name}'s First Holy Communion status?",
            f"Show all sacrament details of {full_name}"
        ]

    elif scope == 'MARRIAGE_STATUS':
        mrg_rec = sac_bundle.get("marriage") if sac_bundle else None
        mrg_date = (mrg_rec.get("mrg_date") if mrg_rec else None) or target_member.get("mrg_date")
        if mrg_date or mrg_rec:
            d_str = str(mrg_date) if mrg_date else "Recorded in parish register"
            spouse = mrg_rec.get("bride_name") if mrg_rec and mrg_rec.get("bride_name") != full_name else (mrg_rec.get("bridegroom_name") if mrg_rec else None)
            spouse_str = f" with **{spouse}**" if spouse else ""
            reply = f"**{full_name}**'s Holy Matrimony was solemnized on `{d_str}`{spouse_str}."
        else:
            reply = f"**{full_name}** has no marriage record in {parish}."
        suggestions = [
            f"Who are {full_name}'s family members?",
            f"Show all sacrament details of {full_name}",
            f"What is {full_name}'s baptism status?"
        ]

    elif scope == 'DEATH_STATUS':
        dth_rec = sac_bundle.get("death") if sac_bundle else None
        dth_date = (dth_rec.get("death_date") if dth_rec else None)
        if dth_date:
            reply = f"**{full_name}** is recorded as deceased on `{dth_date}`."
        else:
            reply = f"No death record found for **{full_name}**."
        suggestions = [
            f"Who are {full_name}'s family members?",
            f"Give family details of {full_name}"
        ]

    elif scope == 'ALL_SACRAMENTS':
        bap_rec = sac_bundle.get("baptism") if sac_bundle else None
        bapt_date = (bap_rec.get("bapt_date") if bap_rec else None) or target_member.get("bapt_date")
        bap_str = f"Received on `{bapt_date}`" if bapt_date else "Not recorded"
        
        fhc_rec = sac_bundle.get("communion") if sac_bundle else None
        fhc_date = (fhc_rec.get("fhc_date") if fhc_rec else None) or target_member.get("fhc_date")
        fhc_str = f"Received on `{fhc_date}`" if fhc_date else "Not recorded"
        
        cnf_rec = sac_bundle.get("confirmation") if sac_bundle else None
        cnf_date = (cnf_rec.get("cnf_date") if cnf_rec else None) or target_member.get("cnf_date")
        cnf_str = f"Received on `{cnf_date}`" if cnf_date else "Not recorded"
        
        mrg_rec = sac_bundle.get("marriage") if sac_bundle else None
        mrg_date = (mrg_rec.get("mrg_date") if mrg_rec else None) or target_member.get("mrg_date")
        mrg_str = f"Solemnized on `{mrg_date}`" if mrg_date else "Not recorded"
        
        lines = [
            f"### 🕊️ Sacrament details for {full_name}:",
            f"- **Baptism:** {bap_str}",
            f"- **First Holy Communion:** {fhc_str}",
            f"- **Confirmation:** {cnf_str}",
            f"- **Marriage:** {mrg_str}"
        ]
        reply = "\n".join(lines)
        suggestions = [
            f"Who are {full_name}'s family members?",
            f"What is {full_name}'s phone number?",
            f"Where does {full_name} live?"
        ]

    elif scope == 'MEMBER_PHONE':
        mob = target_member.get("mobile")
        if not mob and fam_bundle:
            mob = fam.get("mobile") or fam.get("phone")
        if mob:
            reply = f"**{full_name}**'s mobile number is **{mob}**."
        else:
            reply = f"No phone number is registered for **{full_name}**."
        suggestions = [
            f"Where does {full_name} live?",
            f"Who are the members of {full_name}'s family?",
            f"What is {full_name}'s baptism status?"
        ]

    elif scope == 'MEMBER_ADDRESS':
        street = target_member.get("family_address") or target_member.get("street") or fam.get("street") or ""
        city = fam.get("city") or ""
        address = f"{street}, {city}".strip(", ") if city else street
        if address:
            reply = f"**{full_name}**'s registered address is:\n{address}{f' (Family Card: `{card_no}`)' if card_no else ''}"
        else:
            reply = f"No address is registered for **{full_name}**."
        suggestions = [
            f"What is {full_name}'s phone number?",
            f"Who are the members of {full_name}'s family?",
            f"What is {full_name}'s baptism status?"
        ]

    elif scope == 'FAMILY_MEMBERS_ONLY':
        members = (fam_bundle.get("members") if fam_bundle else None) or [target_member]
        lines = [f"### 👨‍👩‍👧‍👦 Family members of {full_name} (Family Card: {card_no}):\n"]
        lines.append("| # | Member Name | Relationship | Gender | Age | Contact |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for idx, m in enumerate(members, 1):
            rel = m.get("relationship_id") or ("Head of Family" if str(m.get("is_family_head")) in ["1", "True", "true"] else "-")
            age_str = str(m.get("age")) if m.get("age") and m.get("age") > 0 else "-"
            mob_str = m.get("mobile") or "-"
            lines.append(f"| {idx} | **{m.get('full_name')}** | {rel} | {m.get('gender') or '-'} | {age_str} | {mob_str} |")
        reply = "\n".join(lines)
        suggestions = [
            f"What is {full_name}'s phone number?",
            f"Give family details of {full_name}",
            f"What is {full_name}'s baptism status?"
        ]

    elif scope == 'FAMILY_DETAILS':
        fam_id = fam.get("name") or target_member.get("family_id") or "N/A"
        fam_name = fam.get("reference") or f"Family of {full_name}"
        anbiyam = fam.get("parish_bcc_id") or "N/A"
        street = fam.get("street") or "N/A"
        city = fam.get("city") or ""
        address = f"{street}, {city}".strip(", ") if city else street
        phone = fam.get("mobile") or fam.get("phone") or target_member.get("mobile") or "N/A"
        members = (fam_bundle.get("members") if fam_bundle else None) or []
        head_mem = next((m for m in members if str(m.get("is_family_head")) in ["1", "True", "true"] or m.get("relationship_id") in ["Head of Family", "Husband"]), None)
        family_head = head_mem.get("full_name") if head_mem else fam_name

        lines = [
            f"### 🏠 FAMILY: {card_no}",
            f"- **Family Card Number:** `{card_no}`",
            f"- **Family Name:** {fam_name}",
            f"- **Family Head:** {family_head}",
            f"- **BCC / Anbiyam:** {anbiyam}",
            f"- **Address:** {address}",
            f"- **Contact:** {phone}\n",
            "#### 👨‍👩‍👧‍👦 FAMILY MEMBERS:",
            "| # | Full Name | Gender | Age | Contact |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        for idx, m in enumerate(members, 1):
            age_str = str(m.get("age")) if m.get("age") and m.get("age") > 0 else "-"
            mob_str = m.get("mobile") or "-"
            lines.append(f"| {idx} | **{m.get('full_name')}** | {m.get('gender') or '-'} | {age_str} | {mob_str} |")
        reply = "\n".join(lines)
        suggestions = [
            f"Show all sacrament details of {full_name}",
            f"What is {full_name}'s baptism status?",
            f"What is {full_name}'s phone number?"
        ]

    else: # GENERAL_MEMBER
        mob = target_member.get("mobile") or "N/A"
        street = target_member.get("family_address") or target_member.get("street") or fam.get("street") or ""
        city = fam.get("city") or ""
        address = f"{street}, {city}".strip(", ") if city else (street or "N/A")
        lines = [
            f"### 👤 Parishioner: {full_name}",
            f"- **Family Card No:** `{card_no}`",
            f"- **Parish:** {parish}",
            f"- **Contact:** {mob}",
            f"- **Address:** {address}"
        ]
        reply = "\n".join(lines)
        suggestions = [
            f"What is {full_name}'s baptism status?",
            f"Who are the members of {full_name}'s family?",
            f"Show all sacrament details of {full_name}"
        ]

    return reply, suggestions[:3]


def format_family_response(bundle: dict, target_member: dict = None) -> str:
    """
    Formats the authoritative Family details response without sacraments dump.
    """
    fam = bundle.get("family") or {}
    members = bundle.get("members") or []
    target = target_member or (members[0] if members else {})
    reply, _ = render_scoped_response("FAMILY_DETAILS", target, None, bundle)
    return reply


def format_member_sacrament_response(target_member: dict, sac_bundle: dict, fam_bundle: dict = None) -> str:
    """
    Formats sacrament status response for a verified parishioner.
    """
    reply, _ = render_scoped_response("ALL_SACRAMENTS", target_member, sac_bundle, fam_bundle)
    return reply
