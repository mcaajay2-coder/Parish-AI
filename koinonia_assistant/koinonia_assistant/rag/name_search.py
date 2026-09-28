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
    'starting', 'begins', 'parishes', 'diocese', 'vicariate', 'status', 'certificate',
    # Tanglish intent / connector particles (Section 7)
    'kudu', 'sollu', 'kaattu', 'enna', 'oda', 'udaiya', 'ku', 'kku', 'patti', 'patri',
    'tha', 'thaa', 'paru', 'eppo', 'enga', 'yaru', 'yaaru',
    # Follow-up pronouns & anaphoric words (MUST NEVER be extracted as literal person names)
    'them', 'they', 'their', 'theirs', 'him', 'her', 'his', 'he', 'she', 'it', 'this', 'these', 'those',
    'avanga', 'avangala', 'avangaloda', 'avangaluku', 'avangalku', 'avaru', 'avar', 'ivanga', 'ivangala', 'ivangaloda',
    'pannu', 'sol', 'sollu', 'kaatu', 'kaatunga', 'kudunga'
}

# Domain & Intent Spelling/Typo Correction Map (Sections 1, 2, 6, 10)
# Strictly corrects intent/action/Christian vocabulary words without altering person/family names or IDs.
INTENT_TYPO_MAP = {
    # details
    'setails': 'details', 'setail': 'details', 'detals': 'details', 'detal': 'details',
    'deatils': 'details', 'deatil': 'details', 'dtails': 'details', 'detials': 'details',
    'detaills': 'details', 'fetails': 'details', 'cetails': 'details', 'retails': 'details',
    'detailes': 'details', 'detais': 'details', 'detaisl': 'details', 'deatls': 'details',
    # family / families
    'famly': 'family', 'famlily': 'family', 'fmaily': 'family', 'famiy': 'family',
    'familiy': 'family', 'faimly': 'family', 'famli': 'family', 'fammily': 'family',
    'famlies': 'families', 'familes': 'families',
    # baptism
    'baptizm': 'baptism', 'baptisim': 'baptism', 'baptisum': 'baptism', 'babtism': 'baptism',
    'baptsm': 'baptism', 'baptisam': 'baptism', 'baptizms': 'baptisms', 'baptismsm': 'baptism',
    # sacrament / sacraments
    'sacremnt': 'sacrament', 'sacrement': 'sacrament', 'sacramnt': 'sacrament',
    'sacramet': 'sacrament', 'sacremnet': 'sacrament', 'sacramnet': 'sacrament',
    'sacrements': 'sacraments', 'sacramnts': 'sacraments', 'sacremnts': 'sacraments',
    # communion
    'comunion': 'communion', 'communon': 'communion', 'comunon': 'communion',
    'communin': 'communion', 'cummunion': 'communion', 'comminion': 'communion',
    # confirmation
    'confimation': 'confirmation', 'conformation': 'confirmation', 'confirmaton': 'confirmation',
    'confirmtion': 'confirmation', 'confermation': 'confirmation',
    # marriage
    'mariage': 'marriage', 'marrige': 'marriage', 'marraige': 'marriage',
    'marige': 'marriage', 'marragie': 'marriage',
    # status
    'staus': 'status', 'sttus': 'status', 'satus': 'status', 'statu': 'status',
    'statsu': 'status', 'stauts': 'status',
    # members / parish / records / contact / address
    'membr': 'member', 'membrs': 'members', 'mebers': 'members', 'meber': 'member',
    'mmbers': 'members', 'mambers': 'members', 'memebrs': 'members', 'memebers': 'members',
    'parsh': 'parish', 'parsih': 'parish', 'parishoner': 'parishioner', 'parishners': 'parishioners',
    'parishoners': 'parishioners',
    'recrods': 'records', 'reocrds': 'records', 'recrds': 'records', 'recors': 'records',
    'numbr': 'number', 'nmber': 'number', 'nuber': 'number',
    'adress': 'address', 'addres': 'address', 'adres': 'address', 'addrss': 'address',
    'contct': 'contact', 'conatct': 'contact', 'cntact': 'contact',
    'moble': 'mobile', 'moblie': 'mobile',
    'anbiym': 'anbiyam', 'anbyam': 'anbiyam', 'anbiaym': 'anbiyam',
    'dioces': 'diocese', 'diocse': 'diocese', 'diocease': 'diocese',
    'vicariat': 'vicariate', 'vicarate': 'vicariate', 'vicariet': 'vicariate',
    'certifcate': 'certificate', 'certficate': 'certificate', 'cetificate': 'certificate',
}

CANONICAL_INTENT_DOMAINS = (
    'details', 'family', 'families', 'baptism', 'baptisms', 'communion',
    'confirmation', 'marriage', 'sacrament', 'sacraments', 'status',
    'members', 'parishioner', 'parishioners', 'records', 'address',
    'contact', 'certificate', 'anbiyam', 'diocese', 'vicariate', 'registered'
)

PROTECTED_NAME_ROOTS = {
    'roselin', 'roseline', 'roslin', 'rosline', 'raselin', 'karoline', 'caroline',
    'antony', 'anthony', 'antoy', 'antoni', 'selvan', 'selvam', 'joseph', 'mary',
    'maria', 'john', 'peter', 'paul', 'thomas', 'james', 'charles', 'francis',
    'xavier', 'sahayaraj', 'arokiaraj', 'savariraj', 'irudayaraj', 'sebastian',
    'michael', 'david', 'daniel', 'stephen', 'vincent', 'lawrence', 'patrick',
    'martin', 'george', 'robert', 'william', 'henry', 'albert', 'arthur', 'walter',
    'adaikala', 'abinaya', 'anciy', 'asmiya', 'elitia', 'sathiyanathan', 'yelagiri'
}

TANGLISH_PARTICLES = {'kudu', 'sollu', 'kaattu', 'enna', 'oda', 'udaiya', 'ku', 'kku', 'patti', 'patri', 'tha', 'thaa', 'paru'}

KNOWN_BCCS = [
    "Arockiya Annai Anbiyam",
    "St. Francis Xavier Anbiyam",
    "St. Joseph Anbiyam",
    "St. Joseph BCC",
    "Lourdu Matha BCC",
    "Sacred Heart BCC",
    "St. Antony BCC",
    "Infant Jesus BCC",
    "Holy Cross BCC",
    "Christ the King BCC",
    "St. Jude BCC",
    "Mother Teresa BCC",
    "Velankanni Matha BCC",
    "Holy Family BCC",
    "St. Thomas BCC"
]

BCC_ALIASES = {
    "arockiya annai": "Arockiya Annai Anbiyam",
    "arokiya annai": "Arockiya Annai Anbiyam",
    "arockia annai": "Arockiya Annai Anbiyam",
    "arokkiya annai": "Arockiya Annai Anbiyam",
    "ஆரோக்கிய அன்னை": "Arockiya Annai Anbiyam",
    "அரோக்கிய அன்னை": "Arockiya Annai Anbiyam",
    "st. joseph": "St. Joseph Anbiyam",
    "st joseph": "St. Joseph Anbiyam",
    "saint joseph": "St. Joseph Anbiyam",
    "joseph": "St. Joseph Anbiyam",
    "புனித சூசையப்பர்": "St. Joseph Anbiyam",
    "சூசையப்பர்": "St. Joseph Anbiyam",
    "st. francis xavier": "St. Francis Xavier Anbiyam",
    "st francis xavier": "St. Francis Xavier Anbiyam",
    "saint francis xavier": "St. Francis Xavier Anbiyam",
    "francis xavier": "St. Francis Xavier Anbiyam",
    "xavier": "St. Francis Xavier Anbiyam",
    "புனித பிரான்சிஸ் சேவியர்": "St. Francis Xavier Anbiyam",
    "சேவியர்": "St. Francis Xavier Anbiyam",
}

def resolve_canonical_bcc(raw_bcc_text: str, user_parish: str = None) -> Optional[str]:
    """
    Resolves raw user input (e.g. 'arockiya annai', 'St. Francis Xavier Anbiyam', 'arokiya annai anbiyam')
    to the canonical BCC name in tabFamily.
    """
    if not raw_bcc_text:
        return None
    cleaned = re.sub(r'[\.,\-_/\\\'"]', ' ', raw_bcc_text.strip().lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Exact or stripped check against known BCCs
    for bcc in KNOWN_BCCS:
        bcc_norm = re.sub(r'[\.,\-_/\\\'"]', ' ', bcc.lower())
        bcc_norm = re.sub(r'\s+', ' ', bcc_norm).strip()
        if cleaned == bcc_norm:
            return bcc

    # Check alias dictionary
    for alias, canonical in BCC_ALIASES.items():
        if alias in cleaned:
            return canonical

    # Base token matching
    for bcc in KNOWN_BCCS:
        bcc_base = re.sub(r'\b(?:anbiyam|bcc)\b', '', bcc.lower()).strip()
        cleaned_base = re.sub(r'\b(?:anbiyam|bcc)\b', '', cleaned).strip()
        if cleaned_base and (cleaned_base in bcc_base or bcc_base in cleaned_base):
            return bcc

    # Attempt query from database tabFamily
    try:
        import frappe
        if getattr(frappe, "db", None):
            db_res = frappe.db.sql(
                "SELECT DISTINCT parish_bcc_id FROM `tabFamily` WHERE parish_bcc_id LIKE %s LIMIT 1",
                (f"%{raw_bcc_text.strip()}%",),
                as_dict=True
            )
            if db_res and db_res[0].get("parish_bcc_id"):
                return db_res[0]["parish_bcc_id"]
    except Exception:
        pass

    return raw_bcc_text.strip()

def normalize_bcc_for_comparison(bcc_str: str) -> str:
    if not bcc_str:
        return ""
    s = bcc_str.lower().strip()
    s = re.sub(r'[\.,\-_/\\\'"]', ' ', s)
    s = re.sub(r'\b(?:anbiyam|bcc|அன்பியம்)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def validate_candidate_hard_constraints(candidate: dict, constraints: dict) -> tuple[bool, list[str]]:
    """
    STRICT ENTITY, BCC, FAMILY AND MULTI-CONSTRAINT VALIDATION (Rules 3, 4, 5, 10, 16):
    Validates candidate against all explicit user constraints:
    - BCC / Anbiyam
    - Family Card Number
    - Parish Name
    Returns (is_valid, failure_reasons).
    """
    reasons = []

    # 1. BCC / Anbiyam Constraint (Rule 4)
    req_bcc = constraints.get("bcc")
    if req_bcc:
        cand_bcc = candidate.get("anbiyam") or candidate.get("parish_bcc_id") or ""
        norm_req = normalize_bcc_for_comparison(req_bcc)
        norm_cand = normalize_bcc_for_comparison(cand_bcc)

        bcc_matched = False
        if norm_req and norm_cand:
            if norm_req == norm_cand or norm_req in norm_cand or norm_cand in norm_req:
                bcc_matched = True
            elif resolve_canonical_bcc(norm_cand) == resolve_canonical_bcc(norm_req):
                bcc_matched = True

        if not bcc_matched:
            cand_display = cand_bcc or "Unassigned BCC"
            reasons.append(f"registered in '{cand_display}', not '{req_bcc}'")

    # 2. Family Card Constraint (Rule 5)
    req_card = constraints.get("family_card")
    if req_card:
        cand_card = (candidate.get("family_register_number") or candidate.get("family_card") or "").strip().upper()
        norm_req_card = req_card.strip().upper()
        if cand_card != norm_req_card:
            reasons.append(f"registered under family card '{cand_card}', not '{norm_req_card}'")

    # 3. Parish Constraint (Rule 15)
    req_parish = constraints.get("parish")
    if req_parish:
        cand_parish = (candidate.get("parish_id") or "").strip().lower()
        norm_req_p = req_parish.strip().lower()
        if norm_req_p not in cand_parish and cand_parish not in norm_req_p:
            reasons.append(f"registered in '{candidate.get('parish_id')}', not '{req_parish}'")

    is_valid = (len(reasons) == 0)
    return is_valid, reasons

def extract_query_entities_and_constraints(query_text: str, user_parish: str = None) -> dict:
    """
    EXTRACT ALL ENTITIES BEFORE RETRIEVAL (Rule 1 & 2):
    Identifies all entities and filtering constraints:
    - person/member name
    - BCC / Anbiyam
    - family card number
    - member ID
    - parish
    - year/date
    - sacrament
    - requested intent & response scope
    Strips explicit constraints from query before isolating the person name.
    """
    q = (query_text or "").strip()
    constraints = {
        "person_name": None,
        "bcc": None,
        "family_card": None,
        "member_id": None,
        "parish": None,
        "year": None,
        "sacrament": None,
        "intent": "GENERAL_MEMBER",
        "scope": "GENERAL_MEMBER",
        "stripped_query": q
    }
    if not q:
        return constraints

    working_q = q

    # 1. Family Card Constraint
    m_card = re.search(r'\b(?:card[:\s]+|from\s+|in\s+|family\s*card[:\s]+)?([A-Z]{2,5}[/\-]\d{1,5})\b', working_q, re.IGNORECASE)
    if m_card:
        constraints["family_card"] = m_card.group(1).upper()
        working_q = working_q[:m_card.start()] + " " + working_q[m_card.end():]
        working_q = re.sub(r'\s+', ' ', working_q).strip()

    # 2. BCC Constraint
    matched_bcc = None
    for bcc in KNOWN_BCCS:
        pat = r'\b(?:in\s+|from\s+|under\s+|belonging\s+to\s+|at\s+)?' + re.escape(bcc) + r'\b'
        m = re.search(pat, working_q, re.IGNORECASE)
        if m:
            matched_bcc = bcc
            working_q = working_q[:m.start()] + " " + working_q[m.end():]
            working_q = re.sub(r'\s+', ' ', working_q).strip()
            break

    if not matched_bcc:
        for alias, can_bcc in BCC_ALIASES.items():
            pat = r'\b(?:in\s+|from\s+|under\s+|belonging\s+to\s+|at\s+)?' + re.escape(alias) + r'(?:\s+anbiyam|\s+bcc)?\b'
            m = re.search(pat, working_q, re.IGNORECASE)
            if m:
                matched_bcc = can_bcc
                working_q = working_q[:m.start()] + " " + working_q[m.end():]
                working_q = re.sub(r'\s+', ' ', working_q).strip()
                break

    if not matched_bcc:
        m_gen = re.search(r'\b(?:in\s+|from\s+|under\s+|belonging\s+to\s+|at\s+)?([A-Za-z0-9\.\'\s]{3,30}?)\s+(?:anbiyam|bcc)\b', working_q, re.IGNORECASE)
        if m_gen:
            cand_bcc = m_gen.group(1).strip()
            cand_bcc = re.sub(r'^(?:the|our|this|a|an)\s+', '', cand_bcc, flags=re.IGNORECASE).strip()
            matched_bcc = resolve_canonical_bcc(cand_bcc, user_parish)
            working_q = working_q[:m_gen.start()] + " " + working_q[m_gen.end():]
            working_q = re.sub(r'\s+', ' ', working_q).strip()

    if not matched_bcc:
        m_ta = re.search(r'([\u0B80-\u0BFF\s]{3,30}?)\s*(?:அன்பியத்தில்|அன்பியத்தின்|அன்பியம்)', working_q)
        if m_ta:
            cand_ta = m_ta.group(1).strip()
            matched_bcc = resolve_canonical_bcc(cand_ta, user_parish)
            working_q = working_q[:m_ta.start()] + " " + working_q[m_ta.end():]
            working_q = re.sub(r'\s+', ' ', working_q).strip()

    constraints["bcc"] = matched_bcc

    # 3. Parish Constraint
    m_parish = re.search(r'\b(?:in\s+|from\s+|of\s+)?([A-Za-z\s]+?)\s+parish\b', working_q, re.IGNORECASE)
    if m_parish:
        p_name = m_parish.group(1).strip().title()
        if p_name.lower() not in ('our', 'this', 'the'):
            constraints["parish"] = f"{p_name} Parish"
            working_q = working_q[:m_parish.start()] + " " + working_q[m_parish.end():]
            working_q = re.sub(r'\s+', ' ', working_q).strip()

    constraints["stripped_query"] = working_q

    # 4. Intent & Scope extraction
    q_low = working_q.lower()
    if re.search(r'\b(?:how\s+many\s+members|number\s+of\s+members|how\s+many\s+people)\b', q_low) or 'எத்தனை நபர்கள்' in q_low or 'எத்தனை உறுப்பினர்கள்' in q_low:
        constraints["intent"] = "FAMILY_MEMBER_COUNT"
        constraints["scope"] = "FAMILY_MEMBER_COUNT"
    elif re.search(r'\b(?:family\s+details|family\s+members|household)\b', q_low) or 'குடும்ப விவரங்கள்' in q_low:
        constraints["intent"] = "FAMILY_DETAILS"
        constraints["scope"] = "FAMILY_DETAILS"
    elif re.search(r'\b(?:baptism|baptised|baptized)\b', q_low) or 'திருமுழுக்கு' in q_low:
        constraints["intent"] = "BAPTISM_STATUS"
        constraints["scope"] = "BAPTISM_STATUS"
        constraints["sacrament"] = "BAPTISM"
    elif re.search(r'\b(?:communion|fhc)\b', q_low) or 'முதல் நற்கருணை' in q_low:
        constraints["intent"] = "COMMUNION_STATUS"
        constraints["scope"] = "COMMUNION_STATUS"
        constraints["sacrament"] = "COMMUNION"
    elif re.search(r'\b(?:confirmation)\b', q_low) or 'உறுதிப்பூசுதல்' in q_low:
        constraints["intent"] = "CONFIRMATION_STATUS"
        constraints["scope"] = "CONFIRMATION_STATUS"
        constraints["sacrament"] = "CONFIRMATION"
    elif re.search(r'\b(?:marriage|wedding)\b', q_low) or 'திருமணம்' in q_low:
        constraints["intent"] = "MARRIAGE_STATUS"
        constraints["scope"] = "MARRIAGE_STATUS"
        constraints["sacrament"] = "MARRIAGE"
    elif re.search(r'\b(?:mobile|phone|contact)\b', q_low) or 'தொடர்பு எண்' in q_low:
        constraints["intent"] = "MEMBER_PHONE"
        constraints["scope"] = "MEMBER_PHONE"
    elif re.search(r'\b(?:address|where\s+does)\b', q_low) or 'முகவரி' in q_low:
        constraints["intent"] = "MEMBER_ADDRESS"
        constraints["scope"] = "MEMBER_ADDRESS"
    else:
        constraints["intent"] = "GENERAL_MEMBER"
        constraints["scope"] = "GENERAL_MEMBER"

    # 5. Extract Person Name from cleaned remaining working_q
    clean_p = working_q
    clean_p = re.sub(r'[\?!]+$', '', clean_p).strip()

    # Multilingual / Tamil script handling
    if any('\u0B80' <= ch <= '\u0BFF' for ch in clean_p):
        try:
            from koinonia_assistant.rag.tamil_utils import extract_person_entity_from_multilingual_query
            ent = extract_person_entity_from_multilingual_query(clean_p)
            extracted_ta = ent.get("transliterated_name") or ent.get("original_name")
            if extracted_ta:
                constraints["person_name"] = extracted_ta
                return constraints
        except Exception:
            pass

    # Check Form A1: How many people/members are in <person>'s family
    m_fam_cnt = re.search(
        r'\b(?:how\s+many|number\s+of|count\s+of)\s+(?:people|members|persons|family\s+members)?\s*(?:are\s+)?(?:there\s+)?(?:in|of)\s+(.+?)(?:\'s|’s|\s+family|\s+household|\?|$)',
        clean_p,
        re.IGNORECASE
    )
    if m_fam_cnt:
        clean_p = m_fam_cnt.group(1).strip()
    else:
        # Check Form C: 'details of <person>'
        m_of = re.search(r'\b(?:details\s+of|details\s+for|records\s+of|status\s+of|of|for|about)\s+([A-Za-z0-9\.\s]+)$', clean_p, re.IGNORECASE)
        if m_of:
            clean_p = m_of.group(1).strip()
        else:
            # Check Form D/E
            clean_p = re.sub(r'^(?:show\s+|get\s+|find\s+|view\s+|give\s+|tell\s+me\s+about\s+|who\s+is\s+)', '', clean_p, flags=re.IGNORECASE).strip()
            clean_p = re.sub(r'\s+(?:family\s+details?|family\s+members?|family|household|details|status|info|records?)$', '', clean_p, flags=re.IGNORECASE).strip()

    # Final cleanup
    clean_p = re.sub(r'(?:\'s|’s)$', '', clean_p).strip()
    clean_p = re.sub(r'^(?:the|a|an|parishioner|member)\s+', '', clean_p, flags=re.IGNORECASE).strip()
    clean_p = re.sub(r'\s+', ' ', clean_p).strip()

    constraints["person_name"] = clean_p if clean_p else None
    return constraints


def preprocess_user_query(raw_query: str) -> dict:
    """
    Lightweight Query Normalization and Spelling-Correction Preprocessor (Sections 1-13).
    - Corrects spelling/typing/STT mistakes in intent/action/Christian domain words (e.g. 'setails' -> 'details',
      'famly' -> 'family', 'baptizm' -> 'baptism', 'sacremnt' -> 'sacrament').
    - NEVER modifies database identifiers (Family Cards like YLG/004, Member IDs, phone numbers, dates).
    - NEVER over-corrects person/family names ('Antoy Raj S' stays 'Antoy Raj S' in entity_text so
      database-aware name resolution can verify it against the authorized parish scope).
    - Separates `intent_text` from `entity_text` and supports Tamil/Tanglish input.
    """
    orig = (raw_query or "").strip()
    if not orig:
        return {
            "original_query": "",
            "corrected_query": "",
            "intent_text": "",
            "entity_text": "",
            "language": "en",
            "corrections": [],
            "correction_confidence": "high",
        }

    # Detect language: Tamil script vs Tanglish vs English
    has_tamil_script = any('\u0B80' <= ch <= '\u0BFF' for ch in orig)
    normalized_ws = re.sub(r'\s+', ' ', orig).strip()
    raw_tokens = normalized_ws.split(' ')

    corrected_tokens = []
    corrections = []
    confidence = "high"
    has_tanglish = False

    for tok in raw_tokens:
        # Preserve leading/trailing punctuation around token
        m_tok = re.match(r'^([^\w\u0B80-\u0BFF/]*)([\w\u0B80-\u0BFF/\-\.@\']+)([^\w\u0B80-\u0BFF/]*)$', tok)
        if not m_tok:
            corrected_tokens.append(tok)
            continue

        prefix_p, core, suffix_p = m_tok.group(1), m_tok.group(2), m_tok.group(3)
        core_low = core.lower()

        if core_low in TANGLISH_PARTICLES:
            has_tanglish = True

        # Rule 5: NEVER spell-correct identifiers, cards (YLG/004), numbers, phones, emails, or Tamil script tokens
        if (
            any(c.isdigit() for c in core)
            or '/' in core
            or '@' in core
            or '-' in core
            or any('\u0B80' <= c <= '\u0BFF' for c in core)
            or len(core_low) <= 3
            or core_low in PROTECTED_NAME_ROOTS
        ):
            corrected_tokens.append(tok)
            continue

        # Rule 1 & 6: High-confidence dictionary lookup for intent/domain typos
        if core_low in INTENT_TYPO_MAP:
            repl = INTENT_TYPO_MAP[core_low]
            corrections.append({"from": core, "to": repl})
            corrected_tokens.append(f"{prefix_p}{repl}{suffix_p}")
            continue

        # Already a canonical domain/reserved word
        if core_low in RESERVED_GENERIC_WORDS or core_low in CANONICAL_INTENT_DOMAINS:
            corrected_tokens.append(tok)
            continue

        # High-confidence fuzzy check ONLY against canonical intent/domain words (len >= 6, score >= 86.0)
        best_canon = None
        best_score = 0.0
        for canon in CANONICAL_INTENT_DOMAINS:
            if abs(len(core_low) - len(canon)) <= 2:
                sc = fuzz.ratio(core_low, canon)
                if sc > best_score:
                    best_score = sc
                    best_canon = canon

        if best_canon and best_score >= 86.0:
            corrections.append({"from": core, "to": best_canon})
            corrected_tokens.append(f"{prefix_p}{best_canon}{suffix_p}")
            if best_score < 90.0:
                confidence = "medium"
        else:
            # Rule 2 & 9: Preserve entity/unknown words untouched
            corrected_tokens.append(tok)

    corrected_query = " ".join(corrected_tokens)
    lang = "ta" if has_tamil_script else ("tanglish" if has_tanglish else "en")

    # Separate intent_text and entity_text from corrected_query
    clean_no_id = re.sub(r'\s*\((?:Member\s*ID|Family\s*ID|Family\s*Card|Family|ID|Card)[:\s0-9A-Za-z,\s\-/]+\)', '', corrected_query).strip()
    clean_no_id = re.sub(r'[\?!]+$', '', clean_no_id).strip()

    intent_words = []
    entity_words = []
    for idx, w in enumerate(clean_no_id.split()):
        w_clean = w.lower().strip('.,?!\'":;')
        if not w_clean:
            continue
        if idx > 0 and len(w_clean) == 1 and w_clean.isalpha():
            entity_words.append(w.strip('.,?!\'":;'))
        elif w_clean in RESERVED_GENERIC_WORDS or w_clean in CANONICAL_INTENT_DOMAINS:
            if w_clean not in ('of', 'for', 'about', 'the', 'a', 'an', 'please', 'can', 'you', 'tell', 'me', 'give', 'show', 'get', 'find') and w_clean not in TANGLISH_PARTICLES:
                intent_words.append(w_clean)
        else:
            entity_words.append(w.strip('.,?!\'":;'))

    intent_text = " ".join(intent_words).strip() or "general query"
    entity_text = " ".join(entity_words).strip()

    return {
        "original_query": orig,
        "corrected_query": corrected_query,
        "intent_text": intent_text,
        "entity_text": entity_text,
        "language": lang,
        "corrections": corrections,
        "correction_confidence": confidence,
    }


def normalize_name(name_str: str) -> str:
    """
    Normalizes a name for comparison:
    - Lowercase
    - Trim spaces
    - Collapse multiple spaces into one
    - Remove punctuation (.,-_/\\\'"), normalize '.' and other separators
    - Preserve initials (e.g. 'S', 'P', 'B')
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
    - 'FAMILY_MEMBER_COUNT'
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
    prep = preprocess_user_query(query_text)
    corrected_text = prep.get("corrected_query") or query_text
    clean = re.sub(r'\s*\((?:Member\s*ID|Family\s*ID|Family|ID|Card)[:\s0-9A-Za-z,\s\-]+\)', '', corrected_text).strip()
    clean = re.sub(r'[\?!]+$', '', clean).strip()
    low = clean.lower()

    # Detect Family / Household / Family Members terms (WHO = FAMILY_MEMBERS)
    has_fam_word = bool(
        re.search(r'\b(?:family|household|kudumbam|kudumbathil)\b', low)
        or any(k in low for k in ['குடும்பம்', 'குடும்பத்தில்', 'குடும்பத்தின்', 'குடும்ப'])
    )

    # Detect Sacrament terms (WHAT = SACRAMENT_RECORDS / ALL_SACRAMENTS)
    is_all_sacs = bool(
        re.search(r'\b(?:all\s+sacraments?|all\s+sacramental)\b', low)
        or any(k in low for k in ['அனைத்து திருவருட்சாதன', 'அனைத்து அருட்சாதன', 'எல்லா திருவருட்சாதன', 'திருவருட்சாதனங்கள்', 'திருவருட்சாதனங்களை', 'அருட்சாதனங்கள்', 'அருட்சாதனங்களை'])
    )
    has_baptism = bool(re.search(r'\b(?:baptism|baptisms|baptised|baptized)\b', low) or any(k in low for k in ['ஞானஸ்நானம்', 'ஞானஸ்நான', 'திருமுழுக்கு']))
    has_communion = bool(re.search(r'\b(?:communion|first\s+holy\s+communion|fhc|eucharist)\b', low) or any(k in low for k in ['நற்கருணை', 'முதல் நற்கருணை', 'புதுநன்மை', 'முதல் திருவிருந்து']))
    has_confirmation = bool(re.search(r'\b(?:confirmation|confirmations|chrism)\b', low) or any(k in low for k in ['உறுதிப்பூசுதல்', 'உறுதிபூசுதல்']))
    has_marriage = bool(re.search(r'\b(?:marriage|marriages|matrimony|wedding|spouse|married)\b', low) or any(k in low for k in ['திருமணம்', 'விவாகம்']))
    has_death = bool(re.search(r'\b(?:death|deceased|burial|died)\b', low) or any(k in low for k in ['இறப்பு', 'அடக்கம்']))
    has_generic_sac = bool(
        re.search(r'\b(?:sacraments?\s+details?|sacraments?\s+records?|sacramental\s+status|sacraments?|sacrements?)\b', low)
        or any(k in low for k in ['அருட்சாதன', 'திருவருட்சாதன'])
    )
    has_any_sacrament = bool(is_all_sacs or has_baptism or has_communion or has_confirmation or has_marriage or has_death or has_generic_sac)

    # 0A. FAMILY + SACRAMENT COMBINATION QUERY RULE (Highest Semantic Priority):
    # When a query contains BOTH a family reference AND a sacrament reference:
    # The SACRAMENT determines WHAT is requested, and FAMILY determines WHO (every family member).
    if has_fam_word and has_any_sacrament:
        if is_all_sacs:
            return 'FAMILY_ALL_SACRAMENTS'
        if has_baptism:
            return 'FAMILY_BAPTISM_RECORDS'
        if has_communion:
            return 'FAMILY_COMMUNION_RECORDS'
        if has_confirmation:
            return 'FAMILY_CONFIRMATION_RECORDS'
        if has_marriage:
            return 'FAMILY_MARRIAGE_RECORDS'
        if has_death:
            return 'FAMILY_DEATH_RECORDS'
        return 'FAMILY_ALL_SACRAMENTS'

    # 0B. Family Member Count ("Antony Selvan குடும்பம் எத்தனை நபர்கள் உள்ளார்கள்", "How many people are in Antony Selvan's family?")
    has_cnt_word = bool(
        re.search(r'\b(?:how\s+many|number\s+of|count|total|evlo|evalo|ethanai|eththanai)\b', low)
        or any(k in low for k in ['எத்தனை', 'மொத்தம்', 'எண்ணிக்கை'])
    )
    if has_fam_word and has_cnt_word:
        return 'FAMILY_MEMBER_COUNT'

    # 1. Phone / Mobile / Contact
    if bool(re.search(r'\b(?:phone|mobile|contact|cell)\b', low) or any(k in low for k in ['தொலைபேசி', 'அலைபேசி', 'தொடர்பு எண்', 'போன்'])):
        return 'MEMBER_PHONE'

    # 2. Address / Where live
    if bool(re.search(r'\b(?:address|residence)\b', low) or re.search(r'\bwhere\s+does\b.*\b(?:live|reside|stay)\b', low) or any(k in low for k in ['முகவரி', 'எங்கே வசிக்கிறார்', 'எங்கு வசிக்கிறார்'])):
        return 'MEMBER_ADDRESS'

    # 3. Family members only (Section 6: "Antony Selvan குடும்பத்தில் யார் யார் இருக்கிறார்கள்?", "Show Antony Selvan's family members.")
    if bool(
        re.search(r'\b(?:family\s+members?|members\s+of\s+(?:the\s+)?family|members\s+of|yaar\s+yaar|yaar)\b', low)
        or re.search(r'\bwho\s+are\b.*\b(?:members|family|in)\b', low)
        or any(k in low for k in ['குடும்ப உறுப்பினர்கள்', 'குடும்ப உறுப்பினர்', 'குடும்ப அங்கத்தினர்கள்', 'யார் யார்', 'குடும்பத்தில் யார்', 'பட்டியல் இடு', 'பட்டியலிடு', 'உறுப்பினர்களை'])
    ):
        return 'FAMILY_MEMBERS_ONLY'

    # 4. Family details
    if bool(re.search(r'\b(?:family\s+details?|family\s+info(?:rmation)?|family\s+records?|family\s+card|details\s+of\s+family)\b', low) or any(k in low for k in ['குடும்ப விவரம்', 'குடும்ப அட்டை', 'குடும்ப'])):
        return 'FAMILY_DETAILS'

    # 5. Specific Sacraments (Single Person status requested)
    if not is_all_sacs:
        if has_baptism:
            return 'BAPTISM_STATUS'
        if has_communion:
            return 'COMMUNION_STATUS'
        if has_confirmation:
            return 'CONFIRMATION_STATUS'
        if has_marriage:
            return 'MARRIAGE_STATUS'
        if has_death:
            return 'DEATH_STATUS'

    # 6. All Sacraments / Sacrament bundle (Single Person)
    if is_all_sacs or has_generic_sac:
        return 'ALL_SACRAMENTS'

    if any(k in low for k in ['family of', 'household of']):
        return 'FAMILY_DETAILS'

    return 'GENERAL_MEMBER'


def build_candidate_prompt(query_text: str, person_name: str, cand_name: str, card_no: str, scope: str = None) -> tuple[str, str]:
    """
    Preserves the user's original query intent across candidate selection without ever exposing
    internal Member IDs or Family IDs. Returns (prompt_with_card, clean_display_text).
    Supports English, Tamil, and Tanglish query contexts.
    """
    eff_scope = scope or determine_response_scope(query_text)
    clean_cand = re.sub(r'\s+', ' ', cand_name or '').strip()
    clean_card = (card_no or '').strip()
    card_suffix = f" (Card: {clean_card})" if clean_card else ""

    from koinonia_assistant.rag.tamil_utils import is_tamil
    is_ta = is_tamil(query_text)

    if is_ta:
        if eff_scope == 'FAMILY_BAPTISM_RECORDS':
            display_text = f"{clean_cand} குடும்பத்தினரின் திருமுழுக்குப் பதிவுகள்"
        elif eff_scope == 'FAMILY_COMMUNION_RECORDS':
            display_text = f"{clean_cand} குடும்பத்தினரின் முதல் நற்கருணைப் பதிவுகள்"
        elif eff_scope == 'FAMILY_CONFIRMATION_RECORDS':
            display_text = f"{clean_cand} குடும்பத்தினரின் உறுதிப்பூசுதல் பதிவுகள்"
        elif eff_scope == 'FAMILY_MARRIAGE_RECORDS':
            display_text = f"{clean_cand} குடும்பத்தினரின் திருமணப் பதிவுகள்"
        elif eff_scope == 'FAMILY_ALL_SACRAMENTS':
            display_text = f"{clean_cand} குடும்பத்தினரின் அனைத்து அருட்சாதனப் பதிவுகள்"
        elif eff_scope == 'FAMILY_MEMBER_COUNT':
            display_text = f"{clean_cand} குடும்பத்தில் எத்தனை பேர் உள்ளனர்?"
        elif eff_scope in ('FAMILY_DETAILS', 'FAMILY_MEMBERS_ONLY'):
            display_text = f"{clean_cand} குடும்ப விவரங்கள்"
        elif eff_scope == 'BAPTISM_STATUS':
            display_text = f"{clean_cand} திருமுழுக்கு நிலை என்ன?"
        elif eff_scope == 'COMMUNION_STATUS':
            display_text = f"{clean_cand} முதல் நற்கருணை பெற்றாரா?"
        elif eff_scope == 'CONFIRMATION_STATUS':
            display_text = f"{clean_cand} உறுதிப்பூசுதல் பெற்றாரா?"
        elif eff_scope == 'MARRIAGE_STATUS':
            display_text = f"{clean_cand} திருமண நிலை என்ன?"
        elif eff_scope == 'ALL_SACRAMENTS':
            display_text = f"{clean_cand} அருட்சாதன விவரங்கள்"
        elif eff_scope == 'MEMBER_PHONE':
            display_text = f"{clean_cand} தொடர்பு எண் என்ன?"
        elif eff_scope == 'MEMBER_ADDRESS':
            display_text = f"{clean_cand} முகவரி என்ன?"
        else:
            display_text = f"{clean_cand} குடும்ப விவரங்கள்"
    else:
        if eff_scope == 'FAMILY_BAPTISM_RECORDS':
            display_text = f"Show baptism records of {clean_cand}'s family"
        elif eff_scope == 'FAMILY_COMMUNION_RECORDS':
            display_text = f"Show First Holy Communion records of {clean_cand}'s family"
        elif eff_scope == 'FAMILY_CONFIRMATION_RECORDS':
            display_text = f"Show confirmation records of {clean_cand}'s family"
        elif eff_scope == 'FAMILY_MARRIAGE_RECORDS':
            display_text = f"Show marriage records of {clean_cand}'s family"
        elif eff_scope == 'FAMILY_ALL_SACRAMENTS':
            display_text = f"Show all sacrament records of {clean_cand}'s family"
        elif eff_scope == 'FAMILY_MEMBER_COUNT':
            display_text = f"How many people are in {clean_cand}'s family?"
        elif eff_scope == 'FAMILY_DETAILS':
            display_text = f"Show family details of {clean_cand}"
        elif eff_scope == 'FAMILY_MEMBERS_ONLY':
            display_text = f"Show all family members of {clean_cand}"
        elif eff_scope == 'ALL_SACRAMENTS':
            display_text = f"Show sacrament details for {clean_cand}"
        elif eff_scope == 'BAPTISM_STATUS':
            display_text = f"Show baptism status of {clean_cand}"
        elif eff_scope == 'COMMUNION_STATUS':
            display_text = f"Show communion status of {clean_cand}"
        elif eff_scope == 'CONFIRMATION_STATUS':
            display_text = f"Show confirmation status of {clean_cand}"
        elif eff_scope == 'MARRIAGE_STATUS':
            display_text = f"Show marriage status of {clean_cand}"
        elif eff_scope == 'MEMBER_PHONE':
            display_text = f"What is {clean_cand}'s phone number?"
        elif eff_scope == 'MEMBER_ADDRESS':
            display_text = f"What is {clean_cand}'s address?"
        else:
            display_text = f"Show family details of {clean_cand}"

    return f"{display_text}{card_suffix}", display_text


def detect_statistical_query(query_text: str) -> Optional[dict]:
    """
    STATISTICS QUESTIONS MUST BYPASS PERSON RESOLUTION:
    Determines whether a query is asking for an aggregate/statistical result
    (e.g., gender counts, member totals, family totals, BCC distribution, annual sacrament counts).
    Returns structured statistical dimensions with person_name = None, or None if not an aggregate query.
    """
    if not query_text:
        return None
    q = query_text.lower().strip()
    q_clean = re.sub(r'[\?!]+$', '', q).strip()

    # Guard: Individual person family count query like "Antony Selvan's family" or "in Antony Selvan's family"
    m_person_fam = re.search(r"\b(?:of|for|in|about)\s+([A-Za-z\s]+?)(?:'s|’s)\s+(?:family\s+members?|family|household)\b", q_clean)
    if m_person_fam:
        name_seg = m_person_fam.group(1).strip().lower()
        if name_seg not in ('women', 'men', 'woman', 'man', 'our', 'the', 'my'):
            return None

    # Guard: Tamil individual person family query e.g. "அந்தோணி செல்வன் குடும்பத்தில் எத்தனை பேர்"
    if 'குடும்பத்தில்' in q_clean or 'குடும்பத்தின்' in q_clean or 'குடும்பம்' in q_clean:
        words = q_clean.split()
        if len(words) >= 2:
            first_w = words[0]
            if first_w not in ('பங்கில்', 'எங்கள்', 'நமது', 'மொத்த', 'எத்தனை', 'இந்த'):
                m_p = re.search(r'^\s*([^\s]+(?:\s+[^\s]+)?)\s+(?:குடும்பத்தில்|குடும்பத்தின்|குடும்பம்)', q_clean)
                if m_p:
                    cand = m_p.group(1)
                    if cand not in ('பங்கு', 'பங்கில்', 'மொத்த', 'எத்தனை', 'அனைத்து'):
                        if any(k in q_clean for k in ['எத்தனை பேர்', 'உறுப்பினர்கள் யார்', 'விவரம்']):
                            return None

    # 1. Detect Statistical / Aggregate intent keywords
    has_count_kw = bool(re.search(
        r'\b(?:how\s+many|number\s+of|count\s+of|count|total\s+number|total\s+count|total|how\s+much|breakdown|distribution|ratio|percentage|statistics|stats|summary)\b',
        q_clean
    ) or any(k in q_clean for k in ['எத்தனை', 'மொத்தம்', 'எண்ணிக்கை', 'புள்ளிவிவரம்', 'விகிதம்', 'பகிர்வு', 'கூட்டுத்தொகை', 'வாரியாக']))

    has_year_wise = bool(re.search(r'\b(?:year[\s\-]*wise|by\s+year|yearly|each\s+year|annual|annually)\b', q_clean) or 'ஆண்டு வாரியாக' in q_clean)

    # 2. Detect Gender dimension
    has_women = bool(re.search(r"\b(?:women|woman|female|females|girls|girl|women's|womens)\b", q_clean) or any(k in q_clean for k in ['பெண்கள்', 'பெண்', 'சிறுமிகள்']))
    has_men = bool(re.search(r"\b(?:men|man|male|males|boys|boy|men's|mens)\b", q_clean) or any(k in q_clean for k in ['ஆண்கள்', 'ஆண்', 'சிறுவர்கள்']))
    has_gender_wise = bool(re.search(r'\b(?:gender[\s\-]*wise|by\s+gender|gender\s+count|gender\s+breakdown|gender)\b', q_clean) or 'பாலின வாரியாக' in q_clean or 'பாலினம்' in q_clean)

    # 3. Detect BCC / Anbiyam dimension
    has_bcc_wise = bool(re.search(r'\b(?:in\s+each\s+bcc|in\s+each\s+anbiyam|each\s+bcc|each\s+anbiyam|bcc[\s\-]*wise|anbiyam[\s\-]*wise|per\s+bcc|per\s+anbiyam|by\s+bcc|by\s+anbiyam|every\s+bcc|every\s+anbiyam)\b', q_clean) or any(k in q_clean for k in ['அன்பிய வாரியாக', 'ஒவ்வொரு அன்பியத்திலும்', 'அன்பியம் வாரியாக']))
    anbiyam_filter = None
    if not has_bcc_wise:
        m_anb = re.search(r"\b(?:in|at|of|for)\s+([A-Za-z0-9\s\.\'\"]+?)\s+(?:anbiyam|bcc)\b", q_clean)
        if m_anb:
            cand = m_anb.group(1).strip()
            cand_clean = re.sub(r'^(?:the|a|an)\s+', '', cand, flags=re.IGNORECASE).strip()
            if cand_clean and cand_clean.lower() not in ('each', 'every', 'all', 'our', 'this'):
                anbiyam_filter = cand_clean
        else:
            m_ta_anb = re.search(r'([\u0B80-\u0BFF\s]+?)\s+(?:அன்பியத்தில்|அன்பியத்தின்|அன்பியம்)', q_clean)
            if m_ta_anb:
                cand_ta = m_ta_anb.group(1).strip()
                cand_ta_clean = re.sub(r'^(?:ஒவ்வொரு|அனைத்து|எங்கள்|நமது)\s+', '', cand_ta).strip()
                if cand_ta_clean:
                    anbiyam_filter = cand_ta_clean

    # 4. Detect Sacrament dimension
    has_baptism = bool(re.search(r'\b(?:baptism|baptisms|baptised|baptized)\b', q_clean) or any(k in q_clean for k in ['திருமுழுக்கு', 'ஞானஸ்நானம்']))
    has_communion = bool(re.search(r'\b(?:communion|first\s+holy\s+communion|fhc)\b', q_clean) or any(k in q_clean for k in ['முதல் நற்கருணை', 'நற்கருணை', 'புதுநன்மை']))
    has_confirmation = bool(re.search(r'\b(?:confirmation|confirmations)\b', q_clean) or any(k in q_clean for k in ['உறுதிப்பூசுதல்', 'உறுதிபூசுதல்']))
    has_marriage = bool(re.search(r'\b(?:marriage|marriages|wedding|weddings)\b', q_clean) or any(k in q_clean for k in ['திருமணம்', 'விவாகம்']))
    has_sacrament = has_baptism or has_communion or has_confirmation or has_marriage or bool(re.search(r'\b(?:sacraments?|sacrements?)\b', q_clean) or 'அருட்சாதன' in q_clean or 'திருவருட்சாதன' in q_clean)

    # Detect Year filter
    m_yr = re.search(r'\b(19\d\d|20\d\d)\b', q_clean)
    year_filter = int(m_yr.group(1)) if m_yr else None

    # 5. Detect Family entity
    has_family = bool(re.search(r'\b(?:families|family\s+count|households|household\s+count)\b', q_clean) or any(k in q_clean for k in ['குடும்பங்கள்', 'குடும்ப எண்ணிக்கை']))

    # 6. Detect Member entity
    has_member = bool(re.search(r'\b(?:members|parishioners|people|persons|population|strength|census)\b', q_clean) or any(k in q_clean for k in ['உறுப்பினர்கள்', 'பங்குமக்கள்', 'மக்கள்']))

    # NOW EVALUATE:
    # A. Gender Statistics
    if (has_count_kw or has_gender_wise) and (has_women or has_men or has_gender_wise):
        if (has_women and has_men) or has_gender_wise:
            group_by = 'GENDER'
            genders = ['FEMALE', 'MALE']
        elif has_women:
            group_by = None
            genders = ['FEMALE']
        else:
            group_by = None
            genders = ['MALE']
        return {
            'is_statistical': True,
            'intent': 'MEMBER_STATISTICS',
            'metric': 'COUNT',
            'entity': 'MEMBER',
            'group_by': group_by,
            'gender': genders,
            'anbiyam': anbiyam_filter,
            'scope': 'AUTHORIZED_PARISH',
            'person_name': None
        }

    # B. BCC / Anbiyam Statistics
    if has_bcc_wise or (has_count_kw and anbiyam_filter):
        entity = 'FAMILY' if has_family and not has_member else 'MEMBER'
        group_by = 'BCC' if has_bcc_wise else None
        return {
            'is_statistical': True,
            'intent': 'MEMBER_STATISTICS' if entity == 'MEMBER' else 'FAMILY_STATISTICS',
            'metric': 'COUNT',
            'entity': entity,
            'group_by': group_by,
            'anbiyam': anbiyam_filter,
            'scope': 'AUTHORIZED_PARISH',
            'person_name': None
        }

    # C. Sacrament Statistics
    if has_sacrament and (has_count_kw or has_year_wise or year_filter):
        sac = 'BAPTISM' if has_baptism else ('COMMUNION' if has_communion else ('CONFIRMATION' if has_confirmation else ('MARRIAGE' if has_marriage else 'SACRAMENT')))
        return {
            'is_statistical': True,
            'intent': 'SACRAMENT_STATISTICS',
            'metric': 'COUNT',
            'entity': sac,
            'group_by': 'YEAR' if has_year_wise else None,
            'year': year_filter,
            'scope': 'AUTHORIZED_PARISH',
            'person_name': None
        }

    # D. Parish Family Count
    if (has_count_kw and has_family) or q_clean in ['total families', 'family count', 'number of families']:
        return {
            'is_statistical': True,
            'intent': 'FAMILY_STATISTICS',
            'metric': 'COUNT',
            'entity': 'FAMILY',
            'group_by': None,
            'scope': 'AUTHORIZED_PARISH',
            'person_name': None
        }

    # E. Parish Member Count
    if (has_count_kw and (has_member or 'in our parish' in q_clean or 'in the parish' in q_clean or 'in this parish' in q_clean)) or q_clean in ['total members', 'member count', 'number of members', 'total registered members']:
        return {
            'is_statistical': True,
            'intent': 'MEMBER_STATISTICS',
            'metric': 'COUNT',
            'entity': 'MEMBER',
            'group_by': None,
            'scope': 'AUTHORIZED_PARISH',
            'person_name': None
        }

    return None


def classify_query_intent(query_text: str) -> dict:
    """
    Classifies query intent and extracts person name & response scope:
    - 'COUNT_MEMBERS' / 'COUNT_FAMILIES'
    - 'MEMBER_STATISTICS' / 'FAMILY_STATISTICS' / 'SACRAMENT_STATISTICS'
    - 'LIST_MEMBERS' / 'LIST_FAMILIES'
    - Family + Sacrament Scopes: 'FAMILY_BAPTISM_RECORDS', 'FAMILY_COMMUNION_RECORDS',
      'FAMILY_CONFIRMATION_RECORDS', 'FAMILY_MARRIAGE_RECORDS', 'FAMILY_DEATH_RECORDS',
      'FAMILY_ALL_SACRAMENTS'
    - Specific scopes: 'FAMILY_MEMBER_COUNT', 'BAPTISM_STATUS', 'CONFIRMATION_STATUS',
      'COMMUNION_STATUS', 'MARRIAGE_STATUS', 'DEATH_STATUS', 'ALL_SACRAMENTS',
      'MEMBER_PHONE', 'MEMBER_ADDRESS', 'FAMILY_MEMBERS_ONLY', 'FAMILY_DETAILS', 'GENERAL_MEMBER'
    """
    prep = preprocess_user_query(query_text)
    q_clean = (prep.get("corrected_query") or query_text).strip()
    q_low = q_clean.lower()

    # =========================================================================
    # RULE: STATISTICS QUESTIONS MUST BYPASS PERSON RESOLUTION
    # Before invoking any person/member name resolver, determine whether the
    # query is asking for an aggregate/statistical result.
    # If statistical: person_name is strictly None, and resolution is bypassed.
    # =========================================================================
    stats_dims = detect_statistical_query(q_clean)
    if stats_dims:
        return {
            "intent": stats_dims["intent"],
            "scope": stats_dims["intent"],
            "person_name": None,
            "is_statistical": True,
            "statistical_dimensions": stats_dims
        }

    # =========================================================================
    # RULE 1 & 2: EXTRACT ALL ENTITIES & CONSTRAINTS BEFORE RETRIEVAL
    # Strips BCC, card, and parish constraints so prepositions do not corrupt the person name!
    # =========================================================================
    constraints = extract_query_entities_and_constraints(q_clean)

    # Determine Scope first so FAMILY_MEMBER_COUNT / FAMILY_SACRAMENT_RECORDS with a person name are never hijacked
    clean_no_id = re.sub(r'\s*\((?:Member\s*ID|Family\s*ID|Family\s*Card|Family|ID|Card)[:\s0-9A-Za-z,\s\-/]+\)', '', constraints.get("stripped_query") or q_clean).strip()
    clean_no_id = re.sub(r'[\?!\.]+$', '', clean_no_id).strip()
    scope = constraints.get("scope") or determine_response_scope(clean_no_id)
    if scope == "GENERAL_MEMBER":
        det_scope = determine_response_scope(clean_no_id)
        if det_scope != "GENERAL_MEMBER":
            scope = det_scope

    # Extract Person Name
    pname = constraints.get("person_name")

    if not pname:
        # Form A0: '<anything> (of|for|in|about) <person>'s family / <person> family (members)'
        m_of_fam = re.search(
            r'\b(?:of|for|in|about)\s+([A-Za-z\.\s]+?)(?:\'s|’s)?\s+(?:family\s+members?|family|household)\b',
            clean_no_id,
            re.IGNORECASE,
        )
        if m_of_fam:
            pname = m_of_fam.group(1).strip()

    # Form A0b: 'family members in <name>' / 'list the family members in <name>' (name trails after 'in', no possessive)
    if not pname:
        m_fam_in = re.search(
            r'\b(?:family\s+members?|members?|household)\s+(?:of\s+|for\s+|in\s+|about\s+)([A-Za-z\.\s]{3,40})$',
            clean_no_id,
            re.IGNORECASE,
        )
        if m_fam_in:
            pname = m_fam_in.group(1).strip()

    # Form A1: How many people/members are in <person>'s family
    if not pname:
        m_fam_cnt = re.search(
            r'\b(?:how\s+many|number\s+of|count\s+of)\s+(?:people|members|persons|family\s+members)?\s*(?:are\s+)?(?:there\s+)?(?:in|of)\s+(.+?)(?:\'s|’s|\s+family|\s+household)\b',
            clean_no_id,
            re.IGNORECASE,
        )
        if m_fam_cnt:
            pname = m_fam_cnt.group(1).strip()

    # Form A2: Possessive '<prefix> <person>'s <attribute>'
    if not pname:
        m_poss = re.search(
            r'^(?:what\s+is|who\s+are|how\s+many\s+people\s+are\s+in|tell\s+me\s+about|give\s+me|show\s+me|show|get|find|view)?\s*(?:all\s+)?(?:sacrament\s+records?\s+of|baptism\s+records?\s+of|confirmation\s+records?\s+of|communion\s+records?\s+of|marriage\s+records?\s+of|members?\s+of)?\s*([A-Za-z\.\s]+?)(?:\'s|’s)\s+(?:mobile|phone|contact|address|family|baptism|confirmation|communion|first\s+holy\s+communion|marriage|death|sacrament|details|records?|status|info)',
            clean_no_id,
            re.IGNORECASE,
        )
        if m_poss:
            pname = m_poss.group(1).strip()

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

    # Form D: Prefix '<person> (family details|sacrament details|baptism status|...)' (supports 2 to 6+ word names with initials)
    if not pname:
        m_suffix = re.search(
            r'^(?:show\s+|get\s+|find\s+|view\s+|give\s+)?([A-Za-z\.\s]+?)\s+(?:oda\s+|ku\s+|kku\s+)?(?:family\s+details?|family\s+members?|family\s+card|family\s+info|family|sacraments?\s+details?|sacraments?\s+records?|sacramental\s+status|sacraments?|baptism\s+status|baptism\s+details?|baptism\s+records?|confirmation\s+status|confirmation\s+details?|communion\s+status|communion\s+details?|marriage\s+status|marriage\s+details?|phone\s+number|mobile\s+number|contact\s+details?|contact|address)\b',
            clean_no_id,
            re.IGNORECASE
        )
        if m_suffix:
            pname = m_suffix.group(1).strip()

    # Form E: Direct search 'show/find/view/get/who is <person>'
    if not pname:
        m_direct = re.search(r'^(?:show|find|view|get|who\s+is|tell\s+me\s+about)\s+(.+)$', clean_no_id, re.IGNORECASE)
        if m_direct:
            pname = m_direct.group(1).strip()

    # Form F: Multilingual / Tamil / Tanglish / Mixed Entity Extraction (Section 13, 14, 15)
    if not pname:
        try:
            from koinonia_assistant.rag.tamil_utils import extract_person_entity_from_multilingual_query
            ent = extract_person_entity_from_multilingual_query(clean_no_id)
            pname = ent.get("transliterated_name") or ent.get("original_name")
        except Exception:
            pass

    # Form G: Short direct query
    if not pname and len(clean_no_id.split()) <= 5 and not any(w in clean_no_id.lower() for w in ['how', 'what', 'why', 'when', 'where', 'list', 'show', 'total', 'count']):
        pname = clean_no_id

    # Validation & stripping of any leftover generic/intent words while preserving single-letter surname initials
    if pname:
        # If 'of' or 'for' or 'in' remained inside pname (e.g. "baptism records of Antony Selvan"), take the segment after it
        if re.search(r'\b(?:of|for|in|about)\s+', pname, re.IGNORECASE):
            pname = re.split(r'\b(?:of|for|in|about)\s+', pname, flags=re.IGNORECASE)[-1].strip()
        pname = re.sub(r'^(?:the|a|an|parishioner|member|show|get|find|view|give|all)\s+', '', pname, flags=re.IGNORECASE).strip()
        pname = re.sub(r'(?:\'s|’s)$', '', pname).strip()
        raw_words = pname.split()
        filtered_words = []
        for idx, w in enumerate(raw_words):
            w_clean = re.sub(r'(?:\'s|’s)$', '', w).lower().strip('.,?!\'":;-')
            if not w_clean or w_clean.isdigit():
                continue
            # Preserve single-letter surname initials (including 'A') when following a name token
            if idx > 0 and len(w_clean) == 1 and w_clean.isalpha():
                filtered_words.append(re.sub(r'(?:\'s|’s)$', '', w).strip('.,?!\'":;-'))
            elif w_clean not in RESERVED_GENERIC_WORDS:
                filtered_words.append(re.sub(r'(?:\'s|’s)$', '', w).strip('.,?!\'":;-'))
        if not filtered_words:
            pname = None
        else:
            pname = " ".join(filtered_words)

    # 1. Parish-wide COUNT INTENT (only when NO specific person name is present)
    if not pname:
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

        # 2. Parish-wide LIST INTENT (only when NO specific person name is present)
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

    intent = scope if pname else ('GENERAL_QUERY' if scope == 'GENERAL_MEMBER' else scope)
    constraints["person_name"] = pname
    constraints["scope"] = scope
    constraints["intent"] = intent

    return {
        'intent': intent,
        'scope': scope,
        'person_name': pname,
        'constraints': constraints
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


def execute_parish_statistics(
    stats_dims: dict,
    user_parish: str = None,
    user_diocese: str = None,
    auth_ctx: dict = None,
    language: str = "en"
) -> dict:
    """
    Executes parish aggregate/statistical queries directly without invoking person name resolution.
    Handles:
    - Gender-wise member statistics (Male vs Female counts, percentages, table)
    - Single gender queries (e.g. men count, women count)
    - BCC / Anbiyam distribution and member counts
    - Specific Anbiyam gender filtering (e.g. "How many men are there in Arockiya Annai Anbiyam?")
    - Parish member and family totals
    - Sacrament event totals (e.g. baptisms in 2024)
    """
    import frappe
    if not getattr(frappe, "db", None):
        try:
            frappe.connect()
        except Exception:
            pass

    is_ta = language == "ta"
    scope_name = user_parish or user_diocese or "your authorized parish"
    entity = stats_dims.get("entity", "MEMBER")
    group_by = stats_dims.get("group_by")
    genders = stats_dims.get("gender")
    anbiyam = stats_dims.get("anbiyam")
    year = stats_dims.get("year")

    # Base WHERE clauses strictly scoped to authorized parish
    where_m = []
    params_m = []
    where_f = []
    params_f = []

    if user_parish:
        where_m.append("(m.parish_id = %s OR m.parish_id LIKE %s)")
        params_m.extend([user_parish, f"%{user_parish}%"])
        where_f.append("(f.parish_id = %s OR f.parish_id LIKE %s)")
        params_f.extend([user_parish, f"%{user_parish}%"])
    elif user_diocese and user_diocese != "All Dioceses":
        where_m.append("m.diocese_id = %s")
        params_m.append(user_diocese)
        where_f.append("f.diocese_id = %s")
        params_f.append(user_diocese)

    # 1. Gender Statistics (group_by == "GENDER")
    if entity == "MEMBER" and group_by == "GENDER":
        where_clauses = list(where_m)
        params = list(params_m)
        if anbiyam:
            where_clauses.append("f.parish_bcc_id LIKE %s")
            params.append(f"%{anbiyam}%")
        base_w = " AND ".join(where_clauses) if where_clauses else "1=1"
        sql = f"""
            SELECT 
                CASE 
                    WHEN LOWER(m.gender) IN ('female', 'woman', 'women', 'girl') THEN 'Female'
                    WHEN LOWER(m.gender) IN ('male', 'man', 'men', 'boy') THEN 'Male'
                    ELSE 'Other/Unspecified'
                END AS `Gender`,
                COUNT(*) AS `Count`
            FROM `tabMember` m
            LEFT JOIN `tabFamily` f ON m.family_id = f.name
            WHERE {base_w}
            GROUP BY `Gender`
            ORDER BY `Count` DESC
        """
        rows = frappe.db.sql(sql, tuple(params), as_dict=True)
        total_m = sum(r["Count"] for r in rows)
        f_count = next((r["Count"] for r in rows if r["Gender"] == "Female"), 0)
        m_count = next((r["Count"] for r in rows if r["Gender"] == "Male"), 0)

        anb_label = f" in **{anbiyam}**" if anbiyam else ""
        anb_label_ta = f" (**{anbiyam}**)" if anbiyam else ""

        if is_ta:
            lines = [
                f"### 📊 {scope_name}{anb_label_ta} — பாலின வாரியான உறுப்பினர் புள்ளிவிவரம்\n",
                "| பாலினம் | எண்ணிக்கை | சதவீதம் |",
                "| :--- | :--- | :--- |",
                f"| **பெண்கள்** | {f_count:,} | {(f_count / total_m * 100):.1f}% |" if total_m else "| **பெண்கள்** | 0 | 0.0% |",
                f"| **ஆண்கள்** | {m_count:,} | {(m_count / total_m * 100):.1f}% |" if total_m else "| **ஆண்கள்** | 0 | 0.0% |",
                f"| **மொத்த உறுப்பினர்கள்** | **{total_m:,}** | **100.0%** |\n",
                f"**{scope_name}** பங்கில்{anb_label_ta} மொத்தம் **{f_count} பெண்களும்** மற்றும் **{m_count} ஆண்களும்** பதிவு செய்யப்பட்டுள்ளனர் (மொத்தம்: **{total_m} உறுப்பினர்கள்**)."
            ]
        else:
            lines = [
                f"### 📊 {scope_name}{anb_label} — Gender-wise Member Statistics\n",
                "| Gender | Count | Percentage |",
                "| :--- | :--- | :--- |",
                f"| **Female (Women)** | {f_count:,} | {(f_count / total_m * 100):.1f}% |" if total_m else "| **Female (Women)** | 0 | 0.0% |",
                f"| **Male (Men)** | {m_count:,} | {(m_count / total_m * 100):.1f}% |" if total_m else "| **Male (Men)** | 0 | 0.0% |",
                f"| **Total Members** | **{total_m:,}** | **100.0%** |\n",
                f"In **{scope_name}**{anb_label}, there are currently **{f_count} women** and **{m_count} men** registered (Total: **{total_m} members**)."
            ]
        return {
            "reply": "\n".join(lines),
            "generated_sql": sql,
            "data": rows,
            "record_count": total_m,
            "suggested_questions": [
                f"What is the total number of families in {scope_name}?",
                "How many members are in each BCC?",
                f"How many baptisms happened in 2024?"
            ]
        }

    # 2. Single Gender Query (e.g. "How many men are there in Arockiya Annai Anbiyam?", "How many women in our parish?")
    if entity == "MEMBER" and genders and len(genders) == 1:
        target_g = genders[0].capitalize()
        g_val = 'male' if target_g == 'Male' else 'female'
        g_display = 'men' if target_g == 'Male' else 'women'
        g_display_ta = 'ஆண்கள்' if target_g == 'Male' else 'பெண்கள்'

        where_clauses = list(where_m)
        params = list(params_m)
        where_clauses.append("LOWER(m.gender) = %s")
        params.append(g_val)
        if anbiyam:
            where_clauses.append("f.parish_bcc_id LIKE %s")
            params.append(f"%{anbiyam}%")
        base_w = " AND ".join(where_clauses) if where_clauses else "1=1"
        sql = f"""
            SELECT COUNT(m.name) AS count
            FROM `tabMember` m
            LEFT JOIN `tabFamily` f ON m.family_id = f.name
            WHERE {base_w}
        """
        cnt = frappe.db.sql(sql, tuple(params))[0][0]
        actual_anbiyam = anbiyam
        if anbiyam:
            try:
                db_bcc = frappe.db.sql(
                    "SELECT DISTINCT parish_bcc_id FROM `tabFamily` WHERE parish_bcc_id LIKE %s AND parish_bcc_id != '' LIMIT 1",
                    (f"%{anbiyam}%",),
                    as_dict=True
                )
                if db_bcc and db_bcc[0].get("parish_bcc_id"):
                    actual_anbiyam = db_bcc[0]["parish_bcc_id"]
                elif not actual_anbiyam.lower().endswith("anbiyam") and not actual_anbiyam.lower().endswith("bcc"):
                    actual_anbiyam = f"{actual_anbiyam.title()} Anbiyam"
            except Exception:
                actual_anbiyam = anbiyam.title() if anbiyam else ""

        anb_label = f" in **{actual_anbiyam}**" if actual_anbiyam else ""
        anb_label_ta = f", **{actual_anbiyam}** அன்பியத்தில்" if actual_anbiyam else ""
        if is_ta:
            reply = f"**{scope_name}** பங்கில்{anb_label_ta} மொத்தம் **{cnt} {g_display_ta}** பதிவு செய்யப்பட்டுள்ளனர்."
        else:
            reply = f"There are currently **{cnt} {g_display}** registered{anb_label} in **{scope_name}**."
        return {
            "reply": reply,
            "generated_sql": sql,
            "data": [{"Category": f"{target_g} ({scope_name})", "Count": cnt}],
            "record_count": cnt,
            "suggested_questions": [
                "Give the gender-wise member count",
                "How many members are in each BCC?",
                "What is the total number of families?"
            ]
        }

    # 3. BCC / Anbiyam-wise Distribution (group_by == "BCC")
    if group_by == "BCC":
        base_w = " AND ".join(where_f) if where_f else "1=1"
        sql = f"""
            SELECT 
                COALESCE(NULLIF(f.parish_bcc_id, ''), 'Unassigned') AS `anbiyam`,
                COUNT(m.name) AS `member_count`,
                COUNT(DISTINCT f.name) AS `family_count`
            FROM `tabFamily` f
            LEFT JOIN `tabMember` m ON m.family_id = f.name
            WHERE {base_w}
            GROUP BY f.parish_bcc_id
            ORDER BY `member_count` DESC
        """
        rows = frappe.db.sql(sql, tuple(params_f), as_dict=True)
        tot_members = sum(r["member_count"] for r in rows)
        tot_families = sum(r["family_count"] for r in rows)

        if is_ta:
            lines = [
                f"### 📊 {scope_name} — அன்பிய வாரியான உறுப்பினர்கள் மற்றும் குடும்பங்கள் விவரம்\n",
                "| # | அன்பியம் (BCC) | உறுப்பினர்கள் | குடும்பங்கள் |",
                "| :--- | :--- | :--- | :--- |"
            ]
            for idx, r in enumerate(rows, 1):
                lines.append(f"| {idx} | **{r['anbiyam']}** | {r['member_count']} | {r['family_count']} |")
            lines.append(f"| | **மொத்தம்** | **{tot_members}** | **{tot_families}** |\n")
            lines.append(f"**{scope_name}** பங்கில் உள்ள அன்பியங்கள் வாரியான விவரம் மேலே அட்டவணையில் கொடுக்கப்பட்டுள்ளது.")
        else:
            lines = [
                f"### 📊 {scope_name} — BCC / Anbiyam-wise Member & Family Distribution\n",
                "| # | BCC / Anbiyam | Members | Families |",
                "| :--- | :--- | :--- | :--- |"
            ]
            for idx, r in enumerate(rows, 1):
                lines.append(f"| {idx} | **{r['anbiyam']}** | {r['member_count']} | {r['family_count']} |")
            lines.append(f"| | **Total** | **{tot_members}** | **{tot_families}** |\n")
            lines.append(f"Here is the BCC/Anbiyam-wise member and family distribution for **{scope_name}**.")

        return {
            "reply": "\n".join(lines),
            "generated_sql": sql,
            "data": rows,
            "record_count": tot_members,
            "suggested_questions": [
                "Give the gender-wise member count",
                "How many members are in our parish?",
                "What is the total number of families?"
            ]
        }

    # 4. Total Families
    if entity == "FAMILY" and not group_by and not anbiyam:
        return handle_count_families(user_parish=user_parish, user_diocese=user_diocese)

    # 5. Total Members
    if entity == "MEMBER" and not group_by and not anbiyam:
        return handle_count_members(user_parish=user_parish, user_diocese=user_diocese)

    # 6. Sacrament Counts (e.g. Baptisms in 2024)
    if entity in ("BAPTISM", "COMMUNION", "CONFIRMATION", "MARRIAGE", "SACRAMENT"):
        from koinonia_assistant.rag.analytics_engine import SACRAMENT_METRICS
        m_key = entity.lower()
        meta = SACRAMENT_METRICS.get(m_key, SACRAMENT_METRICS["baptism"])
        tbl = meta["table"]
        d_col = meta["date_col"]
        p_cols = meta["parish_cols"]
        label = meta["label"]
        label_ta = meta["ta_label"]

        where_c = ["1=1"]
        params = []
        if year:
            where_c.append(f"YEAR(`{d_col}`) = %s")
            params.append(year)
        if user_parish:
            p_or = " OR ".join([f"`{c}` = %s OR `{c}` LIKE %s" for c in p_cols])
            where_c.append(f"({p_or})")
            for _ in p_cols:
                params.extend([user_parish, f"%{user_parish}%"])

        sql = f"SELECT COUNT(*) AS total_count FROM `{tbl}` WHERE {' AND '.join(where_c)}"
        cnt = frappe.db.sql(sql, tuple(params))[0][0]
        yr_str = f" in **{year}**" if year else ""
        yr_str_ta = f" **{year}**-ல்" if year else ""
        if is_ta:
            reply = f"**{scope_name}** பங்கில்{yr_str_ta} மொத்தம் **{cnt:,}** {label_ta} பதிவுகள் உள்ளன."
        else:
            reply = f"In **{scope_name}**{yr_str}, there are **{cnt:,}** {label.lower()} record(s) registered."
        return {
            "reply": reply,
            "generated_sql": sql,
            "data": [{"Category": f"{label} ({scope_name})", "Year": year or "All", "Count": cnt}],
            "record_count": cnt,
            "suggested_questions": [
                f"Show {label.lower()} counts year-wise",
                "How many members are in our parish?",
                "What is the total number of families?"
            ]
        }

    # Fallback to general member count
    return handle_count_members(user_parish=user_parish, user_diocese=user_diocese)


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
    (r'\brosline\b', 'roselin'),
    (r'\broseline\b', 'roselin'),
    (r'\broslin\b', 'roselin'),
    (r'\bcaroline\b', 'karoline'),
    (r'\bkarolin\b', 'karoline'),
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
            init_penalty = 16.0
        elif not m_inits:
            init_penalty = 6.0

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
    user_vicariate: str = None,
    input_mode: str = "chat",
    original_transcript: str = None
) -> dict:
    """
    High-Precision Member and Family Resolver:
    - CHAT MODE (input_mode == "chat"): Preserves existing top-candidate retrieval process untouched.
    - VOICE MODE (input_mode == "voice"): Applies multi-signal competitor-aware voice name resolution
      layer with token-aware matching (never compact/spaceless alone), scope enforcement, and
      disambiguation cards without data leaks (Rules 1-20).
    """
    if not query_text or not query_text.strip():
        return {"status": "empty"}

    intent_res = classify_query_intent(query_text)
    intent = intent_res.get("intent", "GENERAL_QUERY").lower()
    person_name = intent_res.get("person_name") or ""
    scope = intent_res.get("scope", "GENERAL_MEMBER")
    constraints = intent_res.get("constraints") or extract_query_entities_and_constraints(query_text, user_parish)

    if not person_name:
        return {"status": "not_found", "intent": intent, "response_scope": scope, "input_mode": input_mode}

    # Reject if person_name is only generic words
    tokens = [w.lower().strip('.') for w in person_name.split()]
    if not tokens or all(w in RESERVED_GENERIC_WORDS or w.isdigit() for w in tokens):
        return {"status": "not_found", "intent": intent, "response_scope": scope, "input_mode": input_mode}

    norm_query = normalize_name(person_name)
    if len(norm_query) < 2:
        return {"status": "not_found", "intent": intent, "response_scope": scope, "input_mode": input_mode}

    # 1. Build Pre-Search Jurisdiction Filter (Permission filtering BEFORE matching - Rule 5 & 15)
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
        return {"status": "error", "error": str(e), "intent": intent, "response_scope": scope, "input_mode": input_mode}

    if not members:
        return {"status": "not_found", "intent": intent, "response_scope": scope, "input_mode": input_mode}

    # Check if user clicked [Select] on a candidate card with explicit (Card: YLG/...)
    explicit_card_m = re.search(r'\bCard:\s*([A-Z]{2,5}/\d{1,5})\b', query_text, re.IGNORECASE)
    explicit_card = explicit_card_m.group(1).upper() if explicit_card_m else None

    # Deduplicate members by member_id and score all authorized members against hard constraints
    seen_member_ids = set()
    valid_scored = []
    failed_scored = []
    exact_matches = []

    for m in members:
        if m.member_id in seen_member_ids:
            continue
        seen_member_ids.add(m.member_id)
        m.full_name = re.sub(r'\s+', ' ', m.full_name or '').strip()
        norm_full = normalize_name(m.full_name)
        m_card = (m.family_register_number or '').strip().upper()

        score, category = compute_member_similarity(norm_query, norm_full)
        is_valid, reasons = validate_candidate_hard_constraints(m, constraints)

        if is_valid:
            if explicit_card:
                if m_card == explicit_card:
                    exact_matches.append(m)
            elif norm_full == norm_query:
                exact_matches.append(m)
            valid_scored.append((score, category, m))
        else:
            failed_scored.append((score, category, m, reasons))

    valid_scored.sort(key=lambda x: x[0], reverse=True)
    failed_scored.sort(key=lambda x: x[0], reverse=True)

    # =========================================================================
    # RULE 3, 4, 11: HARD CONSTRAINT ENFORCEMENT & NO SILENT FALLBACK
    # If no candidate satisfied all hard constraints, or if the best-matching person for the name
    # failed a hard constraint while only a poor/fuzzy competitor in another BCC/card matched:
    # =========================================================================
    should_report_constraint_failure = False
    if not valid_scored:
        if failed_scored and failed_scored[0][0] >= 60.0:
            should_report_constraint_failure = True
    elif failed_scored and failed_scored[0][0] >= 85.0:
        top_valid_score = valid_scored[0][0]
        top_failed_score = failed_scored[0][0]
        if top_valid_score < 80.0 or (top_failed_score - top_valid_score) > 15.0:
            should_report_constraint_failure = True

    if should_report_constraint_failure:
        top_failed_score, top_failed_cat, top_failed_m, top_failed_reasons = failed_scored[0]
        from koinonia_assistant.rag.tamil_utils import is_tamil
        is_ta = is_tamil(query_text)
        cand_name = top_failed_m.full_name
        reasons_str_en = ", and ".join(top_failed_reasons)
        if is_ta:
            reasons_ta = []
            for r in top_failed_reasons:
                if "registered under family card" in r:
                    reasons_ta.append(f"குடும்ப அட்டை எண் `{top_failed_m.family_register_number}` கீழ் பதிவு செய்யப்பட்டுள்ளார், `{constraints.get('family_card')}` அல்ல")
                elif "registered in" in r and "not" in r:
                    reasons_ta.append(f"'{top_failed_m.anbiyam}' அன்பியத்தில் பதிவு செய்யப்பட்டுள்ளார், '{constraints.get('bcc')}' அன்பியத்தில் அல்ல")
                else:
                    reasons_ta.append(r)
            reasons_str_ta = " மற்றும் ".join(reasons_ta)
            reply = f"நீங்கள் குறிப்பிட்ட அனைத்து விவரங்களுக்கும் பொருந்தும் பதிவு கிடைக்கவில்லை. **{cand_name}** கண்டறியப்பட்டார், ஆனால் அவர் {reasons_str_ta}."
        else:
            if any("registered under family card" in r for r in top_failed_reasons):
                reply = f"I couldn't find a record matching all the details you provided. **{cand_name}** is {reasons_str_en}."
            else:
                reply = f"I couldn't find a record matching all the details you provided. I found **{cand_name}**, but they are {reasons_str_en}."

        return {
            "status": "constraint_failed",
            "reply": reply,
            "person_name": cand_name,
            "failed_reasons": top_failed_reasons,
            "intent": intent,
            "response_scope": scope,
            "input_mode": input_mode,
            "top_score": top_failed_score
        }

    if not valid_scored:
        return {"status": "not_found", "intent": intent, "response_scope": scope, "input_mode": input_mode}

    all_scored = valid_scored
    top_score, top_cat, top_m = all_scored[0] if all_scored else (0.0, "NONE", None)
    second_score, second_cat, second_m = all_scored[1] if len(all_scored) > 1 else (0.0, "NONE", None)
    score_gap = round(top_score - second_score, 1)
    exact_match_count = len(exact_matches)

    # =========================================================================
    # DECISION LAYER: CHAT VS VOICE DIFFERENTIATION (Rule 1)
    # =========================================================================
    cands_pool = []
    if explicit_card:
        # Rule 6, 7, 10, 18: Exact Family Card / Member ID is authoritative
        card_matches = [m for m in valid_scored if (m.family_register_number or '').strip().upper() == explicit_card]
        if card_matches:
            selected_member = card_matches[0]
            decision = "EXACT_CARD_MATCH"
            action = "DIRECT_RESULT"
        else:
            decision = "NO_MATCH"
            action = "NO_MATCH"
            selected_member = None
    elif input_mode != "voice":
        # =====================================================================
        # RULE 1: DO NOT CHANGE THE EXISTING CHAT RETRIEVAL
        # If input_mode = "chat", continue using the existing chat retrieval process
        # =====================================================================
        if exact_match_count >= 1:
            decision = "EXACT_MATCH"
            action = "DIRECT_RESULT"
            selected_member = exact_matches[0]
        elif top_score >= 68.0 and top_m is not None:
            decision = "TOP_CANDIDATE_AUTO_SELECTED"
            action = "DIRECT_RESULT"
            selected_member = top_m
        else:
            decision = "NO_MATCH"
            action = "NO_MATCH"
            selected_member = None
    else:
        # =====================================================================
        # VOICE-BASED PERSON NAME RESOLUTION AND DISAMBIGUATION (Rules 1 - 20)
        # =====================================================================
        if not all_scored or top_score < 65.0:
            # Rule 16-D: Low Confidence
            decision = "LOW_CONFIDENCE"
            action = "LOW_CONFIDENCE"
            selected_member = None
        else:
            # Rule 4: NEVER USE COMPACT NAME ALONE (Token-aware vs compound matching)
            # Rule 8 & 17: AMBIGUOUS MATCH & ANTI-WRONG-PERSON RULE
            # Detect whether top candidate has close competitors in the authorized parish
            competitors = []
            top_norm = normalize_name(top_m.full_name)
            top_spaceless = top_norm.replace(" ", "")
            top_phon_spaceless = phonetic_normalize(top_spaceless)
            q_tokens, q_base, q_inits = split_base_and_initials(norm_query)

            for cand_score, cand_cat, cand_m in all_scored[1:]:
                if cand_m.member_id == top_m.member_id:
                    continue
                if cand_score < 65.0:
                    break

                cand_norm = normalize_name(cand_m.full_name)
                cand_spaceless = cand_norm.replace(" ", "")
                cand_phon_spaceless = phonetic_normalize(cand_spaceless)

                is_competitor = False

                # Condition 1: Compound / Spaceless / Phonetic Base Conflict (Rule 4 & 8)
                # "Antony Raj S" vs "Antonyraj S" both become "antonyrajs" after removing spaces.
                # Token boundaries must be preserved: ['antony', 'raj'] vs ['antonyraj'].
                # Compact space-free matching alone must NEVER make a silent decision!
                if top_spaceless == cand_spaceless or top_phon_spaceless == cand_phon_spaceless:
                    is_competitor = True

                # Condition 2: Close scoring competitor (Rule 17: Candidate A=94, Candidate B=93)
                elif cand_score >= 75.0 and (top_score - cand_score) <= 12.0:
                    is_competitor = True

                # Condition 3: Prefix token ambiguity (e.g. single-token query "Antony")
                elif (top_cat == "PREFIX_TOKEN" or len(q_tokens) <= 1) and (top_score - cand_score) <= 5.0:
                    is_competitor = True

                if is_competitor:
                    competitors.append((cand_score, cand_cat, cand_m))

            if len(exact_matches) > 1 or competitors:
                # Rule 8, 9, 16-C: AMBIGUOUS -> Return 2-3 suggestion cards without leaking private data
                decision = "AMBIGUOUS"
                action = "SHOW_CANDIDATES"
                cands_pool = [top_m]
                for c_score, c_cat, c_m in competitors:
                    if c_m.member_id not in [x.member_id for x in cands_pool]:
                        cands_pool.append(c_m)
                    if len(cands_pool) >= 3:
                        break
            elif top_score >= 70.0 and top_m is not None:
                # Rule 7 & 16-A/B: High confidence or medium confidence but unique
                # One clearly matching candidate with no meaningful competitor -> Direct retrieval
                decision = "CLEAR_UNIQUE_MATCH"
                action = "DIRECT_RESULT"
                selected_member = top_m
            else:
                decision = "LOW_CONFIDENCE"
                action = "LOW_CONFIDENCE"
                selected_member = None

    # Mandatory Debug Output
    print("=" * 60)
    print(f"ORIGINAL QUERY: {query_text} [input_mode={input_mode}]")
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
        card_str = selected_member.family_register_number or selected_member.family_id or ""
        anbiyam_str = selected_member.anbiyam or (fam_bundle.get("family") or {}).get("parish_bcc_id") or ""
        return {
            "status": "exact",
            "matched_member": selected_member,
            "family_bundle": fam_bundle,
            "sacrament_bundle": sac_bundle,
            "intent": intent,
            "response_scope": scope,
            "member_id": selected_member.member_id,
            "family_id": selected_member.family_id,
            "family_card": card_str,
            "parish_id": selected_member.parish_id,
            "anbiyam": anbiyam_str,
            "match_decision": decision,
            "match_status": decision,
            "confirmation_required": False,
            "top_score": top_score,
            "input_mode": input_mode
        }

    if action == "SHOW_CANDIDATES":
        if input_mode == "voice":
            cand_list = cands_pool[:3]
        elif exact_match_count > 1:
            cand_list = [exact_matches[0]]
        else:
            cand_list = [all_scored[0][2]] if all_scored else []

        top_candidates = []
        for m in cand_list:
            card_str = m.family_register_number or m.family_id
            cand_prompt, cand_display = build_candidate_prompt(query_text, person_name, m.full_name, card_str, scope=scope)
            m_score = next((s for s, c, cand in all_scored if cand.member_id == m.member_id), top_score)
            top_candidates.append({
                "type": "member",
                "member_id": m.member_id,
                "full_name": m.full_name,
                "family_id": m.family_id,
                "family_card": card_str,
                "card_no": card_str,
                "family_name": m.family_name or "Family",
                "anbiyam": m.anbiyam or "",
                "place": m.family_address or m.parish_id,
                "parish_id": m.parish_id,
                "prompt": cand_prompt,
                "display_text": cand_display,
                "similarity": round(m_score, 1)
            })

        c_count = len(top_candidates)
        count_str_en = "two" if c_count == 2 else ("three" if c_count == 3 else f"{c_count}")
        disambig_message = f"I found {count_str_en} closely matching names. Please select the person you mean."

        return {
            "status": "candidates",
            "candidates": top_candidates,
            "disambiguation_message": disambig_message,
            "intent": intent,
            "response_scope": scope,
            "match_decision": decision,
            "match_status": "AMBIGUOUS" if input_mode == "voice" else "FUZZY_MATCH",
            "confirmation_required": True,
            "input_mode": input_mode,
            "top_score": top_score
        }

    if action == "LOW_CONFIDENCE":
        from koinonia_assistant.rag.tamil_utils import is_tamil
        is_ta = is_tamil(query_text)
        reply = (
            "நபரை உறுதியாக அடையாளம் காண முடியவில்லை. முழுப் பெயர் அல்லது குடும்ப அட்டை எண்ணைக் கூறவும்."
            if is_ta
            else "I couldn't identify the person confidently. Please say the full name or family card number."
        )
        return {
            "status": "low_confidence",
            "reply": reply,
            "intent": intent,
            "response_scope": scope,
            "match_decision": decision,
            "match_status": "LOW_CONFIDENCE",
            "input_mode": input_mode,
            "top_score": top_score
        }

    return {"status": "not_found", "intent": intent, "response_scope": scope, "match_decision": decision, "input_mode": input_mode}

def fetch_full_family_bundle(family_id: str, parish_id: str = None) -> dict:
    """
    Authoritative Family & Member Retrieval:
    Uses member.family_id -> tabFamily.name = family_id
    Then retrieves all members where family_id = family_id within authorized parish_id.
    Also enriches every family member with their sacrament bundle so FAMILY_SACRAMENT_RECORDS
    and FAMILY_ALL_SACRAMENTS queries have complete member-by-member sacrament records.
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

    card_no = fam.get("family_register_number") or family_id
    parish = parish_id or fam.get("parish_id")

    if parish:
        members = frappe.db.sql(
            """
            SELECT 
                name as member_id,
                first_name, middle_name, last_name,
                TRIM(CONCAT_WS(' ', first_name, middle_name, last_name)) as full_name,
                gender, mobile, email, dob, age, relationship_id, is_family_head,
                bapt_date, fhc_date, cnf_date, mrg_date, parish_id
            FROM `tabMember`
            WHERE family_id = %s AND (parish_id = %s OR parish_id LIKE %s)
            ORDER BY FIELD(relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, name ASC
            """,
            (family_id, parish, f"%{parish}%"),
            as_dict=True
        ) or []
    else:
        members = frappe.db.sql(
            """
            SELECT 
                name as member_id,
                first_name, middle_name, last_name,
                TRIM(CONCAT_WS(' ', first_name, middle_name, last_name)) as full_name,
                gender, mobile, email, dob, age, relationship_id, is_family_head,
                bapt_date, fhc_date, cnf_date, mrg_date, parish_id
            FROM `tabMember`
            WHERE family_id = %s
            ORDER BY FIELD(relationship_id, 'Head of Family', 'Husband', 'Wife', 'Father', 'Mother', 'Son', 'Daughter') ASC, name ASC
            """,
            (family_id,),
            as_dict=True
        ) or []

    for _m in members:
        _m["full_name"] = re.sub(r'\s+', ' ', _m.get("full_name") or "").strip()
        # Alias relationship_id → relationship so it survives the _id-strip filter in run_query
        _m["relationship"] = _m.get("relationship_id") or (
            "Head of Family" if str(_m.get("is_family_head")) in ["1", "True", "true"] else ""
        )
        _m["sacrament_bundle"] = fetch_member_sacrament_bundle(_m.get("member_id"), parish, card_no)

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

def fetch_member_sacrament_bundle(member_id: str, parish_id: str = None, family_card_no: str = None) -> dict:
    """
    Retrieves sacrament register records specifically for a single member within the authorized parish.
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
    eff_parish = parish_id or mem.get("parish_id")
    fam_id = mem.get("family_id")
    card_no = family_card_no or fam_id

    bap = None
    fhc = None
    cnf = None
    mrg = None
    dth = None

    try:
        baps = frappe.db.sql(
            """
            SELECT name, first_name, last_name, bapt_date, bapt_place, bapt_minister, bapt_god_father, bapt_god_mother
            FROM `tabBaptism`
            WHERE (member_id = %s OR (first_name = %s AND (family_card_no = %s OR family_card_no = %s OR last_name = %s)))
              AND (parish_id = %s OR %s IS NULL)
            LIMIT 1
            """,
            (member_id, mem.get("first_name"), card_no, fam_id, mem.get("last_name"), eff_parish, eff_parish),
            as_dict=True
        )
        if baps:
            bap = baps[0]
    except Exception:
        pass

    try:
        fhcs = frappe.db.sql(
            """
            SELECT name, first_name, last_name, fhc_date, fhc_place, fhc_minister
            FROM `tabCommunion`
            WHERE (member_id = %s OR (first_name = %s AND (family_card_no = %s OR family_card_no = %s)))
              AND (parish_id = %s OR %s IS NULL)
            LIMIT 1
            """,
            (member_id, mem.get("first_name"), card_no, fam_id, eff_parish, eff_parish),
            as_dict=True
        )
        if fhcs:
            fhc = fhcs[0]
    except Exception:
        pass

    try:
        cnfs = frappe.db.sql(
            """
            SELECT name, first_name, last_name, cnf_date, cnf_place, cnf_minister
            FROM `tabConfirmation`
            WHERE (member_id = %s OR (first_name = %s AND (family_card_no = %s OR family_card_no = %s)))
              AND (parish_id = %s OR %s IS NULL)
            LIMIT 1
            """,
            (member_id, mem.get("first_name"), card_no, fam_id, eff_parish, eff_parish),
            as_dict=True
        )
        if cnfs:
            cnf = cnfs[0]
    except Exception:
        pass

    try:
        mrgs = frappe.db.sql(
            """
            SELECT name, bridegroom_name, bride_name, mrg_date, mrg_place, mrg_minister
            FROM `tabMarriage`
            WHERE ((bridegroom_id = %s OR bride_id = %s) OR ((family_card_no = %s OR family_card_no = %s) AND (bridegroom_name LIKE %s OR bride_name LIKE %s)))
              AND (parish_id = %s OR %s IS NULL)
            LIMIT 1
            """,
            (member_id, member_id, card_no, fam_id, f"%{mem.get('first_name')}%", f"%{mem.get('first_name')}%", eff_parish, eff_parish),
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


def build_family_sacrament_response(
    scope: str,
    target_member: dict,
    fam_bundle: dict,
    language: str = "en",
) -> tuple[str, list[str], list[dict]]:
    """
    Executes the FAMILY + SACRAMENT COMBINATION QUERY RULE:
    - Combines every member of the resolved family with their sacrament records.
    - Preserves missing records as "No <Sacrament> record found" (NEVER removes a family member,
      NEVER converts missing data to "No", NEVER invents a date).
    - Returns (reply_markdown, suggested_questions, structured_data_rows).
    """
    is_ta = (language == "ta")
    raw_fn = target_member.get("full_name") or target_member.get("first_name") or "Parishioner"
    full_name = re.sub(r'\s+', ' ', str(raw_fn)).strip()
    parish = target_member.get("parish_id") or "Yelagiri Parish"
    fam = (fam_bundle.get("family") if fam_bundle else None) or {}
    card_no = fam.get("family_register_number") or target_member.get("family_card") or target_member.get("family_id") or "N/A"
    members = (fam_bundle.get("members") if fam_bundle else None) or [target_member]

    if scope == "FAMILY_ALL_SACRAMENTS":
        if is_ta:
            lines = [
                f"### 🕊️ அனைத்து திருவருட்சாதனப் பதிவுகள் — {full_name} குடும்பம் (குடும்ப அட்டை: `{card_no}`)\n",
                "| குடும்ப உறுப்பினர் (Family Member) | திருமுழுக்கு (Baptism) | முதல் நற்கருணை (First Communion) | உறுதிப்பூசுதல் (Confirmation) | திருமணம் (Marriage) |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        else:
            lines = [
                f"### 🕊️ ALL SACRAMENT RECORDS — {full_name.upper()} FAMILY (Family Card: `{card_no}`)\n",
                "| Family Member | Baptism | First Holy Communion | Confirmation | Marriage |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]

        rows = []
        for m in members:
            m_name = m.get("full_name") or m.get("first_name") or "Member"
            sb = m.get("sacrament_bundle") or fetch_member_sacrament_bundle(m.get("member_id"), parish, card_no)
            b_rec = sb.get("baptism")
            fhc_rec = sb.get("communion")
            cnf_rec = sb.get("confirmation")
            mrg_rec = sb.get("marriage")

            b_dt = (b_rec.get("bapt_date") if b_rec else None) or m.get("bapt_date")
            fhc_dt = (fhc_rec.get("fhc_date") if fhc_rec else None) or m.get("fhc_date")
            cnf_dt = (cnf_rec.get("cnf_date") if cnf_rec else None) or m.get("cnf_date")
            mrg_dt = (mrg_rec.get("mrg_date") if mrg_rec else None) or m.get("mrg_date")

            b_val = (f"பதிவு செய்யப்பட்டுள்ளது ({b_dt})" if b_dt else "பதிவு செய்யப்பட்டுள்ளது") if (b_dt or b_rec) and is_ta else (
                (f"Baptized ({b_dt})" if b_dt else "Baptized") if (b_dt or b_rec) else ("திருமுழுக்குப் பதிவு இல்லை" if is_ta else "No Baptism record found")
            )
            fhc_val = (f"பதிவு செய்யப்பட்டுள்ளது ({fhc_dt})" if fhc_dt else "பதிவு செய்யப்பட்டுள்ளது") if (fhc_dt or fhc_rec) and is_ta else (
                (f"Received ({fhc_dt})" if fhc_dt else "Received") if (fhc_dt or fhc_rec) else ("முதல் நற்கருணைப் பதிவு இல்லை" if is_ta else "No Communion record found")
            )
            cnf_val = (f"பதிவு செய்யப்பட்டுள்ளது ({cnf_dt})" if cnf_dt else "பதிவு செய்யப்பட்டுள்ளது") if (cnf_dt or cnf_rec) and is_ta else (
                (f"Confirmed ({cnf_dt})" if cnf_dt else "Confirmed") if (cnf_dt or cnf_rec) else ("உறுதிப்பூசுதல் பதிவு இல்லை" if is_ta else "No Confirmation record found")
            )
            mrg_val = (f"பதிவு செய்யப்பட்டுள்ளது ({mrg_dt})" if mrg_dt else "பதிவு செய்யப்பட்டுள்ளது") if (mrg_dt or mrg_rec) and is_ta else (
                (f"Married ({mrg_dt})" if mrg_dt else "Married") if (mrg_dt or mrg_rec) else ("திருமணப் பதிவு இல்லை" if is_ta else "No Marriage record found")
            )

            lines.append(f"| **{m_name}** | {b_val} | {fhc_val} | {cnf_val} | {mrg_val} |")
            rows.append({
                "Family Member": m_name,
                "Baptism": b_val,
                "First Holy Communion": fhc_val,
                "Confirmation": cnf_val,
                "Marriage": mrg_val,
            })

        if is_ta:
            suggestions = [
                f"{full_name} குடும்பத்தின் திருமுழுக்குப் பதிவுகளை காட்டவும்",
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} குடும்பத்தில் எத்தனை நபர்கள் உள்ளார்கள்?",
            ]
        else:
            suggestions = [
                f"Show baptism records of {full_name}'s family",
                f"Show confirmation records of {full_name}'s family",
                f"Show all family members of {full_name}",
            ]
        return "\n".join(lines), suggestions, rows

    # Specific Sacrament across all family members
    sac_map = {
        "FAMILY_BAPTISM_RECORDS": {
            "en_title": "BAPTISM RECORDS",
            "ta_title": "திருமுழுக்குப் பதிவுகள் (Baptism Records)",
            "en_col": "Baptism Status",
            "ta_col": "திருமுழுக்கு நிலை (Baptism Status)",
            "bundle_key": "baptism",
            "date_key": "bapt_date",
            "place_key": "bapt_place",
            "en_done": "Baptized",
            "ta_done": "திருமுழுக்கு பெற்றுள்ளார்",
            "en_missing": "No Baptism record found",
            "ta_missing": "திருமுழுக்குப் பதிவு இல்லை (No Baptism record found)",
        },
        "FAMILY_COMMUNION_RECORDS": {
            "en_title": "FIRST HOLY COMMUNION RECORDS",
            "ta_title": "முதல் நற்கருணைப் பதிவுகள் (First Holy Communion Records)",
            "en_col": "First Holy Communion Status",
            "ta_col": "முதல் நற்கருணை நிலை",
            "bundle_key": "communion",
            "date_key": "fhc_date",
            "place_key": "fhc_place",
            "en_done": "Received",
            "ta_done": "முதல் நற்கருணை பெற்றுள்ளார்",
            "en_missing": "No First Holy Communion record found",
            "ta_missing": "முதல் நற்கருணைப் பதிவு இல்லை (No record found)",
        },
        "FAMILY_CONFIRMATION_RECORDS": {
            "en_title": "CONFIRMATION RECORDS",
            "ta_title": "உறுதிப்பூசுதல் பதிவுகள் (Confirmation Records)",
            "en_col": "Confirmation Status",
            "ta_col": "உறுதிப்பூசுதல் நிலை",
            "bundle_key": "confirmation",
            "date_key": "cnf_date",
            "place_key": "cnf_place",
            "en_done": "Confirmed",
            "ta_done": "உறுதிப்பூசுதல் பெற்றுள்ளார்",
            "en_missing": "No Confirmation record found",
            "ta_missing": "உறுதிப்பூசுதல் பதிவு இல்லை (No Confirmation record found)",
        },
        "FAMILY_MARRIAGE_RECORDS": {
            "en_title": "MARRIAGE RECORDS",
            "ta_title": "திருமணப் பதிவுகள் (Marriage Records)",
            "en_col": "Marriage Status",
            "ta_col": "திருமண நிலை",
            "bundle_key": "marriage",
            "date_key": "mrg_date",
            "place_key": "mrg_place",
            "en_done": "Married",
            "ta_done": "திருமணம் பதிவு செய்யப்பட்டுள்ளது",
            "en_missing": "No Marriage record found",
            "ta_missing": "திருமணப் பதிவு இல்லை (No Marriage record found)",
        },
        "FAMILY_DEATH_RECORDS": {
            "en_title": "DEATH / BURIAL RECORDS",
            "ta_title": "இறப்பு / அடக்கப் பதிவுகள்",
            "en_col": "Death / Burial Status",
            "ta_col": "இறப்புப் பதிவு நிலை",
            "bundle_key": "death",
            "date_key": "death_date",
            "place_key": "burial_place",
            "en_done": "Recorded",
            "ta_done": "பதிவு செய்யப்பட்டுள்ளது",
            "en_missing": "No Death record found",
            "ta_missing": "இறப்புப் பதிவு இல்லை (No Death record found)",
        },
    }

    cfg = sac_map.get(scope, sac_map["FAMILY_BAPTISM_RECORDS"])
    if is_ta:
        lines = [
            f"### 🕊️ {cfg['ta_title']} — {full_name} குடும்பம் (குடும்ப அட்டை: `{card_no}`)\n",
            f"| குடும்ப உறுப்பினர் (Family Member) | {cfg['ta_col']} | தேதி (Date) | இடம் (Place) |",
            "| :--- | :--- | :--- | :--- |",
        ]
    else:
        lines = [
            f"### 🕊️ {cfg['en_title']} — {full_name.upper()} FAMILY (Family Card: `{card_no}`)\n",
            f"| Family Member | {cfg['en_col']} | Date | Place |",
            "| :--- | :--- | :--- | :--- |",
        ]

    rows = []
    for m in members:
        m_name = m.get("full_name") or m.get("first_name") or "Member"
        sb = m.get("sacrament_bundle") or fetch_member_sacrament_bundle(m.get("member_id"), parish, card_no)
        s_rec = sb.get(cfg["bundle_key"])
        s_date = (s_rec.get(cfg["date_key"]) if s_rec else None) or m.get(cfg["date_key"])
        s_place = (s_rec.get(cfg["place_key"]) if s_rec else None) or (parish if (s_date or s_rec) else "—")

        if s_date or s_rec:
            status_val = cfg["ta_done"] if is_ta else cfg["en_done"]
            date_val = str(s_date) if s_date else "—"
            place_val = str(s_place) if s_place else "—"
        else:
            status_val = cfg["ta_missing"] if is_ta else cfg["en_missing"]
            date_val = "—"
            place_val = "—"

        lines.append(f"| **{m_name}** | {status_val} | {date_val} | {place_val} |")
        rows.append({
            "Family Member": m_name,
            cfg["en_col"]: status_val,
            "Date": date_val,
            "Place": place_val,
        })

    if is_ta:
        suggestions = [
            f"{full_name} குடும்பத்தின் அனைத்து திருவருட்சாதனப் பதிவுகளையும் காட்டவும்",
            f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
            f"{full_name} குடும்பத்தில் எத்தனை நபர்கள் உள்ளார்கள்?",
        ]
    else:
        suggestions = [
            f"Show all sacrament records of {full_name}'s family",
            f"Show all family members of {full_name}",
            f"How many people are in {full_name}'s family?",
        ]
    return "\n".join(lines), suggestions, rows


def render_scoped_response(
    scope: str,
    target_member: dict,
    sac_bundle: dict = None,
    fam_bundle: dict = None,
    language: str = "en",
) -> tuple[str, list[str]]:
    """
    Renders ONLY what the user asked based on scope and responds in the user's language (Sections 5–7, 19, 20):
    - FAMILY_BAPTISM_RECORDS / FAMILY_COMMUNION_RECORDS / FAMILY_CONFIRMATION_RECORDS /
      FAMILY_MARRIAGE_RECORDS / FAMILY_DEATH_RECORDS / FAMILY_ALL_SACRAMENTS:
      Member-by-member sacrament records for every member of the resolved family.
    - FAMILY_MEMBER_COUNT: ONLY the member count of the resolved family (never sacrament statistics).
    - BAPTISM_STATUS: Only Baptism status sentence/details for the single target person.
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
    if scope in (
        "FAMILY_BAPTISM_RECORDS",
        "FAMILY_COMMUNION_RECORDS",
        "FAMILY_CONFIRMATION_RECORDS",
        "FAMILY_MARRIAGE_RECORDS",
        "FAMILY_DEATH_RECORDS",
        "FAMILY_ALL_SACRAMENTS",
    ):
        reply_md, sugg_list, struct_rows = build_family_sacrament_response(
            scope, target_member, fam_bundle or {}, language=language
        )
        if isinstance(fam_bundle, dict):
            fam_bundle["family_sacrament_rows"] = struct_rows
        return reply_md, sugg_list

    is_ta = (language == "ta")
    raw_fn = target_member.get("full_name") or target_member.get("first_name") or "Parishioner"
    full_name = re.sub(r'\s+', ' ', str(raw_fn)).strip()
    parish = target_member.get("parish_id") or "the parish"
    fam = (fam_bundle.get("family") if fam_bundle else None) or {}
    card_no = fam.get("family_register_number") or target_member.get("family_register_number") or target_member.get("family_card") or target_member.get("family_id") or "N/A"
    anbiyam = target_member.get("anbiyam") or fam.get("parish_bcc_id") or ""
    members = (fam_bundle.get("members") if fam_bundle else None) or [target_member]

    if scope == 'FAMILY_MEMBER_COUNT':
        count_val = len(members)
        bcc_suffix = f", {anbiyam}" if anbiyam else ""
        if is_ta:
            reply = f"**{full_name}** குடும்பத்தில் **{count_val}** உறுப்பினர்கள் உள்ளனர் (குடும்ப அட்டை: `{card_no}`{bcc_suffix})."
            suggestions = [
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} குடும்பத்தின் தொடர்பு எண்ணை காட்டவும்",
                f"{full_name} குடும்ப அட்டை விவரங்களை காட்டவும்",
            ]
        else:
            reply = f"There are **{count_val}** members in **{full_name}**'s family (Family Card: `{card_no}`{bcc_suffix})."
            suggestions = [
                f"Show all family members of {full_name}",
                f"What is {full_name}'s phone number?",
                f"Show family details of {full_name}",
            ]

    elif scope == 'BAPTISM_STATUS':
        bap_rec = sac_bundle.get("baptism") if sac_bundle else None
        bapt_date = (bap_rec.get("bapt_date") if bap_rec else None) or target_member.get("bapt_date")
        if bapt_date or bap_rec:
            d_str = str(bapt_date) if bapt_date else ("பங்குப் பதிவேட்டில் பதிவு செய்யப்பட்டுள்ளது" if is_ta else "Recorded in parish register")
            place = (bap_rec.get("bapt_place") if bap_rec else None) or parish
            minister = bap_rec.get("bapt_minister") if bap_rec else None
            godfather = bap_rec.get("bapt_god_father") if bap_rec else None
            if is_ta:
                lines = [
                    f"**{full_name}** அவர்களின் திருமுழுக்கு (Baptism) நிலை:",
                    f"- **நிலை:** திருமுழுக்கு பெற்றுள்ளார்",
                    f"- **தேதி:** `{d_str}`",
                    f"- **பங்கு / இடம்:** {place}",
                ]
                if minister:
                    lines.append(f"- **திருப்பணியாளர்:** {minister}")
                if godfather:
                    lines.append(f"- **ஞானப் பெற்றோர்:** {godfather}")
            else:
                lines = [
                    f"**{full_name}** has received the Sacrament of Baptism.",
                    f"- **Date:** {d_str}",
                    f"- **Parish / Place:** {place}",
                ]
                if minister:
                    lines.append(f"- **Minister:** {minister}")
                if godfather:
                    lines.append(f"- **Godparent:** {godfather}")
            reply = "\n".join(lines)
        else:
            reply = (
                f"**{full_name}** அவர்களுக்கு {parish} பதிவேட்டில் திருமுழுக்குப் பதிவு எதுவும் இல்லை."
                if is_ta
                else f"**{full_name}** has no baptism record in {parish}."
            )
        if is_ta:
            suggestions = [
                f"{full_name} அவர்களின் உறுதிப்பூசுதல் நிலை என்ன?",
                f"{full_name} அவர்களின் முதல் நற்கருணை நிலை என்ன?",
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
            ]
        else:
            suggestions = [
                f"What is {full_name}'s confirmation status?",
                f"What is {full_name}'s First Holy Communion status?",
                f"Show all sacrament details of {full_name}",
            ]

    elif scope == 'COMMUNION_STATUS':
        fhc_rec = sac_bundle.get("communion") if sac_bundle else None
        fhc_date = (fhc_rec.get("fhc_date") if fhc_rec else None) or target_member.get("fhc_date")
        if fhc_date or fhc_rec:
            d_str = str(fhc_date) if fhc_date else ("பங்குப் பதிவேட்டில் பதிவு செய்யப்பட்டுள்ளது" if is_ta else "Recorded in parish register")
            reply = (
                f"**{full_name}** அவர்களின் முதல் நற்கருணை நிலை:\n- **நிலை:** முதல் நற்கருணை பெற்றுள்ளார்\n- **தேதி:** `{d_str}`"
                if is_ta
                else f"**{full_name}** has received the Sacrament of First Holy Communion.\n- **Date:** {d_str}"
            )
        else:
            reply = (
                f"**{full_name}** அவர்களுக்கு {parish} பதிவேட்டில் முதல் நற்கருணைப் பதிவு எதுவும் இல்லை."
                if is_ta
                else f"**{full_name}** has no First Holy Communion record in {parish}."
            )
        if is_ta:
            suggestions = [
                f"{full_name} அவர்களின் ஞானஸ்நான நிலை என்ன?",
                f"{full_name} அவர்களின் உறுதிப்பூசுதல் நிலை என்ன?",
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
            ]
        else:
            suggestions = [
                f"What is {full_name}'s baptism status?",
                f"What is {full_name}'s confirmation status?",
                f"Show all sacrament details of {full_name}",
            ]

    elif scope == 'CONFIRMATION_STATUS':
        cnf_rec = sac_bundle.get("confirmation") if sac_bundle else None
        cnf_date = (cnf_rec.get("cnf_date") if cnf_rec else None) or target_member.get("cnf_date")
        if cnf_date or cnf_rec:
            d_str = str(cnf_date) if cnf_date else ("பங்குப் பதிவேட்டில் பதிவு செய்யப்பட்டுள்ளது" if is_ta else "Recorded in parish register")
            place = (cnf_rec.get("cnf_place") if cnf_rec else None) or parish
            reply = (
                f"**{full_name}** அவர்களின் உறுதிப்பூசுதல் நிலை:\n- **நிலை:** உறுதிப்பூசுதல் பெற்றுள்ளார்\n- **தேதி:** `{d_str}`\n- **பங்கு / இடம்:** {place}"
                if is_ta
                else f"**{full_name}** has received the Sacrament of Confirmation.\n- **Date:** {d_str}\n- **Parish / Place:** {place}"
            )
        else:
            reply = (
                f"**{full_name}** அவர்களுக்கு {parish} பதிவேட்டில் உறுதிப்பூசுதல் பதிவு எதுவும் இல்லை."
                if is_ta
                else f"**{full_name}** has no confirmation record in {parish}."
            )
        if is_ta:
            suggestions = [
                f"{full_name} அவர்களின் ஞானஸ்நான நிலை என்ன?",
                f"{full_name} அவர்களின் முதல் நற்கருணை நிலை என்ன?",
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
            ]
        else:
            suggestions = [
                f"What is {full_name}'s baptism status?",
                f"What is {full_name}'s First Holy Communion status?",
                f"Show all sacrament details of {full_name}",
            ]

    elif scope == 'MARRIAGE_STATUS':
        mrg_rec = sac_bundle.get("marriage") if sac_bundle else None
        mrg_date = (mrg_rec.get("mrg_date") if mrg_rec else None) or target_member.get("mrg_date")
        if mrg_date or mrg_rec:
            d_str = str(mrg_date) if mrg_date else ("பங்குப் பதிவேட்டில் பதிவு செய்யப்பட்டுள்ளது" if is_ta else "Recorded in parish register")
            spouse = mrg_rec.get("bride_name") if mrg_rec and mrg_rec.get("bride_name") != full_name else (mrg_rec.get("bridegroom_name") if mrg_rec else None)
            spouse_str = (f" (துணைவர்: **{spouse}**)" if is_ta else f" with **{spouse}**") if spouse else ""
            reply = (
                f"**{full_name}** அவர்களின் திருமண விவரம்: `{d_str}`{spouse_str}."
                if is_ta
                else f"**{full_name}**'s Holy Matrimony was solemnized on `{d_str}`{spouse_str}."
            )
        else:
            reply = (
                f"**{full_name}** அவர்களுக்கு {parish} பதிவேட்டில் திருமணப் பதிவு எதுவும் இல்லை."
                if is_ta
                else f"**{full_name}** has no marriage record in {parish}."
            )
        if is_ta:
            suggestions = [
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} அவர்களின் ஞானஸ்நான நிலை என்ன?",
                f"{full_name} குடும்பத்தின் தொடர்பு எண்ணை காட்டவும்",
            ]
        else:
            suggestions = [
                f"Who are {full_name}'s family members?",
                f"Show all sacrament details of {full_name}",
                f"What is {full_name}'s baptism status?",
            ]

    elif scope == 'DEATH_STATUS':
        dth_rec = sac_bundle.get("death") if sac_bundle else None
        dth_date = (dth_rec.get("death_date") if dth_rec else None)
        if dth_date:
            reply = (
                f"**{full_name}** அவர்களின் இறப்புத் தேதி: `{dth_date}`."
                if is_ta
                else f"**{full_name}** is recorded as deceased on `{dth_date}`."
            )
        else:
            reply = (
                f"**{full_name}** அவர்களுக்கான இறப்புப் பதிவு எதுவும் இல்லை."
                if is_ta
                else f"No death record found for **{full_name}**."
            )
        suggestions = (
            [
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} குடும்ப அட்டை விவரங்களை காட்டவும்",
            ]
            if is_ta
            else [
                f"Who are {full_name}'s family members?",
                f"Give family details of {full_name}",
            ]
        )

    elif scope == 'ALL_SACRAMENTS':
        bap_rec = sac_bundle.get("baptism") if sac_bundle else None
        bapt_date = (bap_rec.get("bapt_date") if bap_rec else None) or target_member.get("bapt_date")
        bap_str = (f"பெற்றுள்ளார் (`{bapt_date}`)" if is_ta else f"Received on `{bapt_date}`") if bapt_date else ("பதிவு இல்லை" if is_ta else "Not recorded")
        
        fhc_rec = sac_bundle.get("communion") if sac_bundle else None
        fhc_date = (fhc_rec.get("fhc_date") if fhc_rec else None) or target_member.get("fhc_date")
        fhc_str = (f"பெற்றுள்ளார் (`{fhc_date}`)" if is_ta else f"Received on `{fhc_date}`") if fhc_date else ("பதிவு இல்லை" if is_ta else "Not recorded")
        
        cnf_rec = sac_bundle.get("confirmation") if sac_bundle else None
        cnf_date = (cnf_rec.get("cnf_date") if cnf_rec else None) or target_member.get("cnf_date")
        cnf_str = (f"பெற்றுள்ளார் (`{cnf_date}`)" if is_ta else f"Received on `{cnf_date}`") if cnf_date else ("பதிவு இல்லை" if is_ta else "Not recorded")
        
        mrg_rec = sac_bundle.get("marriage") if sac_bundle else None
        mrg_date = (mrg_rec.get("mrg_date") if mrg_rec else None) or target_member.get("mrg_date")
        mrg_str = (f"நடைபெற்றது (`{mrg_date}`)" if is_ta else f"Solemnized on `{mrg_date}`") if mrg_date else ("பதிவு இல்லை" if is_ta else "Not recorded")
        
        if is_ta:
            lines = [
                f"### 🕊️ {full_name} — திருவருட்சாதன விவரங்கள்:",
                f"- **திருமுழுக்கு (Baptism):** {bap_str}",
                f"- **முதல் நற்கருணை (First Holy Communion):** {fhc_str}",
                f"- **உறுதிப்பூசுதல் (Confirmation):** {cnf_str}",
                f"- **திருமணம் (Marriage):** {mrg_str}",
            ]
            suggestions = [
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} குடும்பத்தின் தொடர்பு எண்ணை காட்டவும்",
                f"{full_name} குடும்ப அட்டை விவரங்களை காட்டவும்",
            ]
        else:
            lines = [
                f"### 🕊️ Sacrament details for {full_name}:",
                f"- **Baptism:** {bap_str}",
                f"- **First Holy Communion:** {fhc_str}",
                f"- **Confirmation:** {cnf_str}",
                f"- **Marriage:** {mrg_str}",
            ]
            suggestions = [
                f"Who are {full_name}'s family members?",
                f"What is {full_name}'s phone number?",
                f"Where does {full_name} live?",
            ]
        reply = "\n".join(lines)

    elif scope == 'MEMBER_PHONE':
        mob = target_member.get("mobile")
        if not mob and fam_bundle:
            mob = fam.get("mobile") or fam.get("phone")
        if mob:
            reply = (
                f"**{full_name}** அவர்களின் தொடர்பு எண்: **{mob}**."
                if is_ta
                else f"**{full_name}**'s mobile number is **{mob}**."
            )
        else:
            reply = (
                f"**{full_name}** அவர்களுக்கான தொடர்பு எண் பதிவேட்டில் இல்லை."
                if is_ta
                else f"No phone number is registered for **{full_name}**."
            )
        suggestions = (
            [
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} குடும்ப அட்டை விவரங்களை காட்டவும்",
                f"{full_name} அவர்களின் ஞானஸ்நான நிலை என்ன?",
            ]
            if is_ta
            else [
                f"Where does {full_name} live?",
                f"Who are the members of {full_name}'s family?",
                f"What is {full_name}'s baptism status?",
            ]
        )

    elif scope == 'MEMBER_ADDRESS':
        street = target_member.get("family_address") or target_member.get("street") or fam.get("street") or ""
        city = fam.get("city") or ""
        address = f"{street}, {city}".strip(", ") if city else street
        if address:
            reply = (
                f"**{full_name}** அவர்களின் முகவரி:\n{address}{f' (குடும்ப அட்டை: `{card_no}`)' if card_no else ''}"
                if is_ta
                else f"**{full_name}**'s registered address is:\n{address}{f' (Family Card: `{card_no}`)' if card_no else ''}"
            )
        else:
            reply = (
                f"**{full_name}** அவர்களுக்கான முகவரி பதிவேட்டில் இல்லை."
                if is_ta
                else f"No address is registered for **{full_name}**."
            )
        suggestions = (
            [
                f"{full_name} குடும்பத்தின் தொடர்பு எண்ணை காட்டவும்",
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} அவர்களின் திருமுழுக்கு நிலை என்ன?",
            ]
            if is_ta
            else [
                f"What is {full_name}'s phone number?",
                f"Who are the members of {full_name}'s family?",
                f"What is {full_name}'s baptism status?",
            ]
        )

    elif scope == 'FAMILY_MEMBERS_ONLY':
        if is_ta:
            lines = [f"### 👨‍👩‍👧‍👦 {full_name} குடும்ப உறுப்பினர்கள் (குடும்ப அட்டை: `{card_no}` — மொத்தம் {len(members)} நபர்கள்):\n"]
            lines.append("| # | உறுப்பினர் பெயர் (Name) | உறவுமுறை (Relationship) | பாலினம் (Gender) | வயது (Age) | தொடர்பு எண் (Contact) |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        else:
            lines = [f"### 👨‍👩‍👧‍👦 Family members of {full_name} (Family Card: {card_no}):\n"]
            lines.append("| # | Member Name | Relationship | Gender | Age | Contact |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for idx, m in enumerate(members, 1):
            rel = m.get("relationship_id") or ("Head of Family" if str(m.get("is_family_head")) in ["1", "True", "true"] else "-")
            age_str = str(m.get("age")) if m.get("age") and m.get("age") > 0 else "-"
            mob_str = m.get("mobile") or "-"
            lines.append(f"| {idx} | **{m.get('full_name')}** | {rel} | {m.get('gender') or '-'} | {age_str} | {mob_str} |")
        reply = "\n".join(lines)
        suggestions = (
            [
                f"{full_name} குடும்பத்தில் எத்தனை நபர்கள் உள்ளார்கள்?",
                f"{full_name} குடும்பத்தின் தொடர்பு எண்ணை காட்டவும்",
                f"{full_name} குடும்ப அட்டை விவரங்களை காட்டவும்",
            ]
            if is_ta
            else [
                f"How many people are in {full_name}'s family?",
                f"What is {full_name}'s phone number?",
                f"Give family details of {full_name}",
            ]
        )

    elif scope == 'FAMILY_DETAILS':
        fam_id = fam.get("name") or target_member.get("family_id") or "N/A"
        fam_name = fam.get("reference") or f"Family of {full_name}"
        anbiyam = fam.get("parish_bcc_id") or "N/A"
        street = fam.get("street") or "N/A"
        city = fam.get("city") or ""
        address = f"{street}, {city}".strip(", ") if city else street
        phone = fam.get("mobile") or fam.get("phone") or target_member.get("mobile") or "N/A"
        head_mem = next((m for m in members if str(m.get("is_family_head")) in ["1", "True", "true"] or m.get("relationship_id") in ["Head of Family", "Husband"]), None)
        family_head = head_mem.get("full_name") if head_mem else fam_name

        if is_ta:
            lines = [
                f"### 🏠 குடும்ப விவரம்: {card_no}",
                f"- **குடும்ப அட்டை எண்:** `{card_no}`",
                f"- **குடும்பப் பெயர்:** {fam_name}",
                f"- **குடும்பத் தலைவர்:** {family_head}",
                f"- **அன்பியம் (BCC):** {anbiyam}",
                f"- **முகவரி:** {address}",
                f"- **தொடர்பு எண்:** {phone}",
                f"- **மொத்த உறுப்பினர்கள்:** `{len(members)}`\n",
                "#### 👨‍👩‍👧‍👦 குடும்ப உறுப்பினர்கள்:",
                "| # | உறுப்பினர் பெயர் (Name) | உறவுமுறை (Relationship) | பாலினம் (Gender) | வயது (Age) | தொடர்பு எண் (Contact) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        else:
            lines = [
                f"### 🏠 FAMILY: {card_no}",
                f"- **Family Card Number:** `{card_no}`",
                f"- **Family Name:** {fam_name}",
                f"- **Family Head:** {family_head}",
                f"- **BCC / Anbiyam:** {anbiyam}",
                f"- **Address:** {address}",
                f"- **Contact:** {phone}\n",
                "#### 👨‍👩‍👧‍👦 FAMILY MEMBERS:",
                "| # | Full Name | Relationship | Gender | Age | Contact |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        for idx, m in enumerate(members, 1):
            rel = m.get("relationship_id") or m.get("relationship") or ("Head of Family" if str(m.get("is_family_head")) in ["1", "True", "true"] else "Not recorded")
            age_str = str(m.get("age")) if m.get("age") and m.get("age") > 0 else "-"
            mob_str = m.get("mobile") or "-"
            lines.append(f"| {idx} | **{m.get('full_name')}** | {rel} | {m.get('gender') or '-'} | {age_str} | {mob_str} |")
        reply = "\n".join(lines)
        suggestions = (
            [
                f"{full_name} குடும்பத்தில் எத்தனை நபர்கள் உள்ளார்கள்?",
                f"{full_name} அவர்களின் ஞானஸ்நான நிலை என்ன?",
                f"{full_name} குடும்பத்தின் தொடர்பு எண்ணை காட்டவும்",
            ]
            if is_ta
            else [
                f"How many people are in {full_name}'s family?",
                f"Show sacrament details for {full_name}",
                f"Show baptism status of {full_name}",
            ]
        )

    else:  # GENERAL_MEMBER
        mob = target_member.get("mobile") or "N/A"
        street = target_member.get("family_address") or target_member.get("street") or fam.get("street") or ""
        city = fam.get("city") or ""
        address = f"{street}, {city}".strip(", ") if city else (street or "N/A")
        bcc_name = target_member.get("anbiyam") or fam.get("parish_bcc_id") or "N/A"
        if is_ta:
            lines = [
                f"### 👤 பங்கு உறுப்பினர்: {full_name}",
                f"- **குடும்ப அட்டை எண்:** `{card_no}`",
                f"- **அன்பியம் (BCC):** {bcc_name}",
                f"- **பங்கு:** {parish}",
                f"- **தொடர்பு எண்:** {mob}",
                f"- **முகவரி:** {address}",
            ]
            suggestions = [
                f"{full_name} குடும்பத்தில் எத்தனை நபர்கள் உள்ளார்கள்?",
                f"{full_name} குடும்ப உறுப்பினர்களின் விவரங்களை காட்டவும்",
                f"{full_name} அவர்களின் திருமுழுக்கு நிலை என்ன?",
            ]
        else:
            lines = [
                f"### 👤 Parishioner: {full_name}",
                f"- **Family Card No:** `{card_no}`",
                f"- **BCC / Anbiyam:** {bcc_name}",
                f"- **Parish:** {parish}",
                f"- **Contact:** {mob}",
                f"- **Address:** {address}",
            ]
            suggestions = [
                f"What is {full_name}'s baptism status?",
                f"Who are the members of {full_name}'s family?",
                f"Show all sacrament details of {full_name}",
            ]
        reply = "\n".join(lines)

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
