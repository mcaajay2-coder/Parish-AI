"""
koinonia_assistant.rag.followup_context

Handles conversational follow-up context, anaphora/pronoun resolution,
and entity inheritance across multi-turn dialogs.

Supports English, Tamil, and Tanglish follow-up queries:
- "list them", "show them", "list members", "show the members"
- "give their details", "show their details"
- "show their baptism records", "their baptism records"
- "அவர்களை பட்டியல் இடு", "அவர்களை காட்டு"
- "அவர்களின் விவரங்களை காட்டு", "அவர்களின் குடும்ப விவரம்"
- "family members list பண்ணு", "members list பண்ணு"
- "avanga details kudu", "avanga details kaatu"
"""

import re
import json
import datetime
from typing import Dict, Any, Optional, Tuple, List

# In-memory thread-safe fallback cache for conversation contexts
_CONVERSATION_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}

# Pronoun & anaphoric tokens across English, Tamil, and Tanglish
FOLLOWUP_PRONOUNS_EN = {
    "them", "they", "their", "theirs", "him", "her", "his", "he", "she"
}

FOLLOWUP_PRONOUNS_TA = {
    "அவர்கள்", "அவர்களை", "அவர்களின்", "அவர்களுடைய",
    "அவர்", "அவரை", "அவரின்", "அவருடைய",
    "இவர்கள்", "இவர்களை", "இவர்களின்", "இவர்களுடைய",
    "இவர்", "இவரை", "இவரின்", "இவருடைய",
}

FOLLOWUP_PRONOUNS_TANGLISH = {
    "avanga", "avangala", "avangaloda", "avangaluku", "avangalku",
    "avaru", "avar", "ivanga", "ivangala", "ivangaloda",
}

ALL_FOLLOWUP_PRONOUNS = FOLLOWUP_PRONOUNS_EN | FOLLOWUP_PRONOUNS_TA | FOLLOWUP_PRONOUNS_TANGLISH


def is_followup_reference(query_text: str) -> Dict[str, Any]:
    """
    Detects if the query is a follow-up reference that inherits the entity from the previous turn.
    Returns:
        {
            "is_followup": bool,
            "action": "LIST_MEMBERS" | "FAMILY_DETAILS" | "SACRAMENT_RECORDS" | "MEMBER_COUNT" | "CONTACT_DETAILS",
            "sacrament": "BAPTISM" | "FIRST_HOLY_COMMUNION" | "CONFIRMATION" | "MARRIAGE" | "DEATH" | "ALL_SACRAMENTS" | None,
            "target_scope": "FAMILY_MEMBERS",
            "pronoun": Optional[str],
            "language": "ta" | "tanglish" | "en"
        }
    """
    q = (query_text or "").strip()
    q_low = q.lower()
    
    # Check if query introduces a brand new explicit family card (e.g. YLG/010) or person name
    m_explicit_card = re.search(r'\b[A-Z]{2,5}/\d{1,5}\b', q, re.IGNORECASE)
    if m_explicit_card:
        return {"is_followup": False}

    # Detect language of the query
    is_ta = any('\u0B80' <= c <= '\u0BFF' for c in q)
    is_tang = any(t in q_low for t in ["avanga", "kudu", "kaatu", "pannu", "sollu", "ivanga", "evlo"])
    lang = "ta" if is_ta else ("tanglish" if is_tang else "en")

    # 1. Sacrament Records Follow-up
    # Examples:
    # "show their baptism records", "their baptism records", "baptism records of them",
    # "அவர்களின் திருமுழுக்கு பதிவுகளை காட்டு", "அவர்களின் ஞானஸ்நான பதிவுகள்", "avanga baptism records kaatu"
    sac_map = {
        "BAPTISM": [
            "baptism", "baptisms", "baptised", "baptized", "திருமுழுக்கு", "ஞானஸ்நானம்", "ஞானஸ்நான", "gnanasnanam", "thirumulukku"
        ],
        "FIRST_HOLY_COMMUNION": [
            "communion", "first communion", "first holy communion", "fhc", "நற்கருணை", "முதல் நற்கருணை", "புதுநன்மை", "narkarunai", "puthunanmai"
        ],
        "CONFIRMATION": [
            "confirmation", "confirmations", "confirmed", "உறுதிபூசுதல்", "உறுதி பூசுதல்", "urudhipoosuthal"
        ],
        "MARRIAGE": [
            "marriage", "marriages", "wedding", "matrimony", "திருமணம்", "கல்யாணம்", "thirumanam"
        ],
        "DEATH": [
            "death", "deaths", "deceased", "burial", "இறப்பு", "அடக்கம்", "irappu"
        ],
        "ALL_SACRAMENTS": [
            "all sacraments", "all sacrament", "all sacramental", "sacraments", "sacrament", "sacrements", "sacrement",
            "அனைத்து திருவருட்சாதன", "திருவருட்சாதனங்கள்", "திருவருட்சாதன", "அருட்சாதன"
        ]
    }

    matched_sac = None
    for s_code, keywords in sac_map.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', q_low) or kw in q:
                matched_sac = s_code
                break
        if matched_sac:
            break

    # Check for pronoun presence
    tokens = [w.strip('.,?!:;"\'') for w in q_low.split()]
    has_pronoun = any(w in ALL_FOLLOWUP_PRONOUNS for w in tokens) or any(p in q for p in FOLLOWUP_PRONOUNS_TA)
    if matched_sac:
        # Check if query names an explicit person (e.g. "Show baptism of Mary Britto", "Mary's baptism")
        has_named_person = False
        m_poss = re.search(r'\b([A-Za-z\.\s]+?)(?:\'s|’s)\s+', q)
        if m_poss:
            toks = [w.strip('.,?!:;"\'') for w in m_poss.group(1).lower().split() if w]
            if not any(w in ALL_FOLLOWUP_PRONOUNS for w in toks):
                has_named_person = True

        m_of = re.search(r'\b(?:of|for)\s+([A-Za-z\.\s]{3,40})$', q, re.IGNORECASE)
        if m_of:
            toks = [w.strip('.,?!:;"\'') for w in m_of.group(1).lower().split() if w]
            if not any(w in ALL_FOLLOWUP_PRONOUNS for w in toks) and not any(w in ["family", "members", "household", "them", "all"] for w in toks):
                has_named_person = True

        if not has_named_person:
            # Requires pronoun OR short ellipsis query (e.g. "their baptism records", "show baptism records")
            if has_pronoun or (len(tokens) <= 4 and not re.search(r'\b(?:in|at|parish|diocese|church|year|annual|total|count)\b', q_low)):
                return {
                    "is_followup": True,
                    "action": "SACRAMENT_RECORDS",
                    "sacrament": matched_sac,
                    "target_scope": "FAMILY_MEMBERS",
                    "pronoun": next((w for w in tokens if w in ALL_FOLLOWUP_PRONOUNS), None),
                    "language": lang
                }

    # 2. Member Count Follow-up
    # Examples: "how many members", "how many people in their family", "count of them", "அவர்களின் குடும்பத்தில் எத்தனை பேர்"
    if re.search(r'\b(?:how\s+many\s+(?:people|members)|member\s+count|count\s+of\s+them)\b', q_low) or any(k in q for k in ["எத்தனை பேர்", "எத்தனை நபர்கள்", "எத்தனை உறுப்பினர்கள்", "evlo per", "ethanai per"]):
        return {
            "is_followup": True,
            "action": "MEMBER_COUNT",
            "sacrament": None,
            "target_scope": "FAMILY_MEMBERS",
            "pronoun": next((w for w in tokens if w in ALL_FOLLOWUP_PRONOUNS), None),
            "language": lang
        }

    # 3. Family Details Follow-up
    # Examples:
    # "give their details", "show their details", "tell their details", "their details", "details of them",
    # "அவர்களின் விவரங்களை காட்டு", "அவர்களின் விவரங்கள்", "அவர்களின் தகவல்கள்",
    # "avanga details kudu", "avanga details kaatu", "avangaloda details", "avanga details"
    fam_details_patterns = [
        r'\b(?:give|show|tell|view|get)?\s*(?:their|of\s+them)\s+(?:family\s+)?details?\b',
        r'\btheir\s+(?:family\s+)?details?\b',
        r'\bdetails\s+of\s+them\b',
        r'\b(?:give|show|tell)?\s*their\s+(?:info|information)\b',
        r'அவர்களின்\s+(?:குடும்ப\s+)?(?:விவரங்களை|விவரங்கள்|விவரம்|தகவல்கள்|தகவல்)',
        r'avanga(?:loda)?\s+(?:family\s+)?details?(?:\s+(?:kudu|kaatu|sol))?',
    ]
    for pat in fam_details_patterns:
        if re.search(pat, q, re.IGNORECASE):
            return {
                "is_followup": True,
                "action": "FAMILY_DETAILS",
                "sacrament": None,
                "target_scope": "FAMILY_MEMBERS",
                "pronoun": next((w for w in tokens if w in ALL_FOLLOWUP_PRONOUNS), None),
                "language": lang
            }

    # Standalone short "details" / "family details"
    if q_low in ["details", "family details", "give details", "show details", "details kudu", "details kaatu", "விவரங்கள்", "விவரங்களை காட்டு"]:
        return {
            "is_followup": True,
            "action": "FAMILY_DETAILS",
            "sacrament": None,
            "target_scope": "FAMILY_MEMBERS",
            "pronoun": None,
            "language": lang
        }

    # 4. List / Show Family Members Follow-up
    # Examples:
    # "list them", "show them", "list members", "show the members", "list the members", "show members",
    # "who are the members", "who are they", "display them", "get them",
    # "அவர்களை பட்டியல் இடு", "அவர்களை பட்டியலிடு", "அவர்களை காட்டு", "அவர்களை காட்டுங்கள்",
    # "குடும்ப உறுப்பினர்கள் யார்", "உறுப்பினர்களை பட்டியல் இடு", "உறுப்பினர்களை காட்டு",
    # "family members list பண்ணு", "family members list pannu", "members list பண்ணு", "members list pannu",
    # "avangala list பண்ணு", "avangala list pannu", "avanga yaaru", "avangala show பண்ணு"
    list_patterns = [
        r'^\s*(?:please\s+)?(?:list|show|view|display|get|tell)\s+(?:them|they|the\s+members|members|family\s+members)\s*$',
        r'^\s*who\s+are\s+(?:they|the\s+members|members)\s*$',
        r'^\s*(?:the\s+)?(?:family\s+)?members\s*$',
        r'^\s*(?:அவர்களை|அவர்கள்)\s+(?:பட்டியல்\s*இடு|பட்டியலிடு|பட்டியலிடுக|காட்டு|காட்டுங்கள்)\s*$',
        r'^\s*(?:குடும்ப\s+)?உறுப்பினர்களை\s+(?:பட்டியல்\s*இடு|பட்டியலிடு|காட்டு|காட்டுங்கள்)\s*$',
        r'^\s*குடும்ப\s+உறுப்பினர்கள்\s+யார்\s*$',
        r'^\s*(?:family\s+)?members\s+list\s+(?:பண்ணு|pannu)\s*$',
        r'^\s*avanga(?:la)?\s+(?:list|show)\s+(?:பண்ணு|pannu|kudu|kaatu)\s*$',
        r'^\s*avanga(?:la)?\s+(?:yaaru|yaar)\s*$',
    ]
    for pat in list_patterns:
        if re.search(pat, q, re.IGNORECASE):
            return {
                "is_followup": True,
                "action": "LIST_MEMBERS",
                "sacrament": None,
                "target_scope": "FAMILY_MEMBERS",
                "pronoun": next((w for w in tokens if w in ALL_FOLLOWUP_PRONOUNS), None),
                "language": lang
            }

    # 5. Contact Follow-up
    # Examples: "give their address", "their phone number", "where do they live", "avanga contact kudu"
    contact_patterns = [
        r'\b(?:their|of\s+them)\s+(?:contact|address|phone|mobile)\b',
        r'\bwhere\s+do\s+they\s+live\b',
        r'அவர்களின்\s+(?:முகவரி|தொடர்பு\s+எண்|போன்\s+எண்)',
        r'avanga(?:loda)?\s+(?:contact|phone|mobile|address)(?:\s+(?:kudu|kaatu))?',
    ]
    for pat in contact_patterns:
        if re.search(pat, q, re.IGNORECASE):
            return {
                "is_followup": True,
                "action": "CONTACT_DETAILS",
                "sacrament": None,
                "target_scope": "FAMILY_MEMBERS",
                "pronoun": next((w for w in tokens if w in ALL_FOLLOWUP_PRONOUNS), None),
                "language": lang
            }

    return {"is_followup": False}


def extract_entity_context_from_history(history: List[Dict[str, str]], default_parish: str = None) -> Dict[str, Any]:
    """
    Extracts resolved entity context (family_card, family_id, person_name, member_id, parish)
    by inspecting prior conversation turns in reverse chronological order.
    """
    if not history:
        return {}

    context: Dict[str, Any] = {
        "family_card": None,
        "family_id": None,
        "person_name": None,
        "member_id": None,
        "parish": default_parish,
        "previous_intent": None,
        "previous_scope": None,
    }

    # Walk history in reverse: inspect recent bot and user turns
    for msg in reversed(history):
        content = (msg.get("content") or "").strip()
        role = (msg.get("role") or "").lower()

        # 1. Family Card (e.g. YLG/005)
        if not context["family_card"]:
            m_card = re.search(r'\b([A-Z]{2,5}/\d{1,5})\b', content)
            if m_card:
                context["family_card"] = m_card.group(1).upper()

        # 2. Family ID from SQL or text (e.g. family_id = '3170' or Family ID: 3170)
        if not context["family_id"]:
            m_fid = re.search(r"family_id\s*=\s*['\"]?(\d+)['\"]?", content, re.IGNORECASE) or \
                    re.search(r"\bFamily\s*ID[:\s]+(\d+)\b", content, re.IGNORECASE)
            if m_fid:
                context["family_id"] = m_fid.group(1)

        # 3. Person Name
        if not context["person_name"]:
            # English pattern: **Antony Selvan P**'s family
            m_p1 = re.search(r"\*\*([A-Za-z\.\s]{3,40}?)\*\*(?:'s|’s)?\s+family", content, re.IGNORECASE)
            # Family Head: **Antony Selvan P** or Family Head: Antony Selvan P
            m_p2 = re.search(r"Family\s+Head:\s*\*\*?([A-Za-z\.\s]{3,40}?)\*\*?", content, re.IGNORECASE)
            # Family members of Antony Selvan P
            m_p3 = re.search(r"Family\s+members\s+of\s+\*\*?([A-Za-z\.\s]{3,40}?)\*\*?", content, re.IGNORECASE)
            # Tamil: Antony Selvan P குடும்ப உறுப்பினர்கள்
            m_p4 = re.search(r"###\s*👨‍👩‍👧‍👦\s*\*\*?([A-Za-z\.\s\u0B80-\u0BFF]{3,40}?)\*\*?\s*குடும்ப", content)
            # Tamil: **Antony Selvan P** குடும்பத்தில்
            m_p5 = re.search(r"\*\*([A-Za-z\.\s\u0B80-\u0BFF]{3,40}?)\*\*\s*குடும்ப", content)

            if m_p1:
                context["person_name"] = m_p1.group(1).strip()
            elif m_p2:
                context["person_name"] = m_p2.group(1).strip()
            elif m_p3:
                context["person_name"] = m_p3.group(1).strip()
            elif m_p4:
                context["person_name"] = m_p4.group(1).strip()
            elif m_p5:
                context["person_name"] = m_p5.group(1).strip()
            elif role == "user":
                # Look at user's previous query if it had an extracted person name
                try:
                    from koinonia_assistant.rag.name_search import extract_person_name_from_query
                    p_name = extract_person_name_from_query(content)
                    if p_name and p_name.lower() not in ALL_FOLLOWUP_PRONOUNS:
                        context["person_name"] = p_name
                except Exception:
                    pass

        # 4. Member ID (e.g. WHERE name = '31701')
        if not context["member_id"]:
            m_mid = re.search(r"\bWHERE\s+m?\.?name\s*=\s*'(\d+)'", content, re.IGNORECASE) or \
                    re.search(r"\bMember\s*ID[:\s]+(\d+)\b", content, re.IGNORECASE)
            if m_mid:
                context["member_id"] = m_mid.group(1)

        # 5. Parish
        if not context["parish"]:
            m_par = re.search(r"parish_id\s*=\s*'([^']+)'", content, re.IGNORECASE)
            if m_par:
                context["parish"] = m_par.group(1)

        # Stop if we already have the essential family/person entity
        if (context["family_card"] or context["family_id"]) and context["person_name"]:
            break

    # Enrich missing context from Frappe DB
    enrich_context_from_db(context)
    return context


def enrich_context_from_db(context: Dict[str, Any]) -> None:
    """Uses database to fill in related family/member/card details if one or more are missing."""
    import frappe
    if not frappe.db:
        return

    try:
        # Case 1: family_card is present, fill family_id, family_name, parish
        if context.get("family_card") and (not context.get("family_id") or not context.get("person_name")):
            card = context["family_card"]
            rows = frappe.db.sql(
                "SELECT name, reference, parish_id FROM `tabFamily` WHERE family_register_number = %s LIMIT 1",
                (card,),
                as_dict=True
            )
            if rows:
                if not context.get("family_id"):
                    context["family_id"] = str(rows[0]["name"])
                if not context.get("person_name") and rows[0].get("reference"):
                    context["person_name"] = rows[0]["reference"].rstrip(".")
                if not context.get("parish") and rows[0].get("parish_id"):
                    context["parish"] = rows[0]["parish_id"]

        # Case 2: family_id is present, fill family_card, family_name, parish
        if context.get("family_id") and (not context.get("family_card") or not context.get("person_name")):
            fid = context["family_id"]
            rows = frappe.db.sql(
                "SELECT family_register_number, reference, parish_id FROM `tabFamily` WHERE name = %s LIMIT 1",
                (fid,),
                as_dict=True
            )
            if rows:
                if not context.get("family_card") and rows[0].get("family_register_number"):
                    context["family_card"] = rows[0]["family_register_number"]
                if not context.get("person_name") and rows[0].get("reference"):
                    context["person_name"] = rows[0]["reference"].rstrip(".")
                if not context.get("parish") and rows[0].get("parish_id"):
                    context["parish"] = rows[0]["parish_id"]

        # Case 3: person_name is present, find their family
        if context.get("person_name") and (not context.get("family_card") or not context.get("family_id")):
            pname = context["person_name"]
            # Lookup member by name
            rows = frappe.db.sql(
                """SELECT m.name as member_id, m.family_id, f.family_register_number, f.reference as family_name, m.parish_id
                   FROM `tabMember` m
                   LEFT JOIN `tabFamily` f ON f.name = m.family_id
                   WHERE CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name) LIKE %s
                   LIMIT 1""",
                (f"%{pname}%",),
                as_dict=True
            )
            if rows:
                if not context.get("member_id"):
                    context["member_id"] = str(rows[0]["member_id"])
                if not context.get("family_id") and rows[0].get("family_id"):
                    context["family_id"] = str(rows[0]["family_id"])
                if not context.get("family_card") and rows[0].get("family_register_number"):
                    context["family_card"] = rows[0]["family_register_number"]
                if not context.get("parish") and rows[0].get("parish_id"):
                    context["parish"] = rows[0]["parish_id"]
    except Exception as e:
        print(f"[FollowupContext] DB enrichment warning: {e}")


def save_conversation_context(cache_key: str, context: Dict[str, Any]) -> None:
    """Saves the latest resolved entity context to memory cache and Frappe cache."""
    if not cache_key or not context:
        return
    # Clean and stamp context
    clean_ctx = {
        "family_card": context.get("family_card"),
        "family_id": str(context.get("family_id")) if context.get("family_id") else None,
        "person_name": context.get("person_name"),
        "member_id": str(context.get("member_id")) if context.get("member_id") else None,
        "parish": context.get("parish"),
        "previous_intent": context.get("previous_intent"),
        "previous_scope": context.get("previous_scope"),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    _CONVERSATION_CONTEXT_CACHE[cache_key] = clean_ctx

    try:
        import frappe
        if hasattr(frappe, "cache") and frappe.cache:
            frappe.cache.hset("koinonia_active_context", cache_key, json.dumps(clean_ctx))
    except Exception:
        pass


def get_conversation_context(cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieves cached entity context from memory or Frappe cache."""
    if not cache_key:
        return None
    # Check memory cache first
    if cache_key in _CONVERSATION_CONTEXT_CACHE:
        return _CONVERSATION_CONTEXT_CACHE[cache_key]

    try:
        import frappe
        if hasattr(frappe, "cache") and frappe.cache:
            data = frappe.cache.hget("koinonia_active_context", cache_key)
            if data:
                ctx = json.loads(data)
                _CONVERSATION_CONTEXT_CACHE[cache_key] = ctx
                return ctx
    except Exception:
        pass
    return None


def synthesize_followup_query(
    original_query: str,
    context: Dict[str, Any],
    followup_meta: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Synthesizes a fully qualified, unambiguous query string and canonical intent structure
    inheriting the previous turn's entity (Family, Member, Parish, Family Card).
    
    Example:
        User says: "list them"
        Context: person_name="Antony Selvan P", family_card="YLG/005", family_id="3170"
        Returns:
            resolved_query = "List the family members of Antony Selvan P (Family Card YLG/005)."
            intent_dict = { ... }
    """
    action = followup_meta.get("action", "LIST_MEMBERS")
    sacrament = followup_meta.get("sacrament")
    lang = followup_meta.get("language", "en")
    
    pname = context.get("person_name") or "Family Head"
    card = context.get("family_card") or ""
    fid = context.get("family_id") or ""
    parish = context.get("parish") or ""

    card_str = f" (Family Card {card})" if card else ""
    card_str_ta = f" (குடும்ப அட்டை: {card})" if card else ""

    resolved_query = original_query
    scope = "FAMILY_MEMBERS_ONLY"
    c_intent = "FAMILY_SEARCH"
    sem_intent = "FAMILY_MEMBER_LIST"
    db_op = "LIST_FAMILY_MEMBERS"
    req_info = "MEMBER_LIST"

    if action == "LIST_MEMBERS":
        scope = "FAMILY_MEMBERS_ONLY"
        c_intent = "FAMILY_SEARCH"
        sem_intent = "FAMILY_MEMBER_LIST"
        db_op = "LIST_FAMILY_MEMBERS"
        req_info = "MEMBER_LIST"
        if lang == "ta":
            resolved_query = f"{pname} குடும்ப உறுப்பினர்கள் பட்டியல்{card_str_ta}."
        else:
            resolved_query = f"List the family members of {pname}{card_str}."

    elif action == "FAMILY_DETAILS":
        scope = "FAMILY_DETAILS"
        c_intent = "FAMILY_SEARCH"
        sem_intent = "FAMILY_DETAILS"
        db_op = "RETRIEVE_FAMILY_BY_ID"
        req_info = "FAMILY_DETAILS"
        if lang == "ta":
            resolved_query = f"{pname} குடும்ப விவரங்களை காட்டு{card_str_ta}."
        else:
            resolved_query = f"List the family details of {pname}{card_str}."

    elif action == "SACRAMENT_RECORDS":
        c_intent = "SACRAMENT_SEARCH"
        sem_intent = "FAMILY_ALL_SACRAMENTS" if sacrament == "ALL_SACRAMENTS" else "FAMILY_SACRAMENT_RECORDS"
        db_op = "LOOKUP_FAMILY_ALL_SACRAMENTS" if sacrament == "ALL_SACRAMENTS" else "LOOKUP_FAMILY_SACRAMENT_RECORDS"
        req_info = "ALL_SACRAMENTS" if sacrament == "ALL_SACRAMENTS" else "SACRAMENT_RECORDS"

        sac_scope_map = {
            "BAPTISM": "FAMILY_BAPTISM_RECORDS",
            "FIRST_HOLY_COMMUNION": "FAMILY_COMMUNION_RECORDS",
            "CONFIRMATION": "FAMILY_CONFIRMATION_RECORDS",
            "MARRIAGE": "FAMILY_MARRIAGE_RECORDS",
            "DEATH": "FAMILY_DEATH_RECORDS",
            "ALL_SACRAMENTS": "FAMILY_ALL_SACRAMENTS"
        }
        scope = sac_scope_map.get(sacrament, "FAMILY_ALL_SACRAMENTS")

        sac_names_en = {
            "BAPTISM": "Baptism",
            "FIRST_HOLY_COMMUNION": "First Holy Communion",
            "CONFIRMATION": "Confirmation",
            "MARRIAGE": "Marriage",
            "DEATH": "Death",
            "ALL_SACRAMENTS": "All sacrament"
        }
        sac_names_ta = {
            "BAPTISM": "திருமுழுக்கு",
            "FIRST_HOLY_COMMUNION": "முதல் நற்கருணை",
            "CONFIRMATION": "உறுதிபூசுதல்",
            "MARRIAGE": "திருமண",
            "DEATH": "இறப்பு",
            "ALL_SACRAMENTS": "அனைத்து திருவருட்சாதன"
        }
        s_name_en = sac_names_en.get(sacrament, "Sacrament")
        s_name_ta = sac_names_ta.get(sacrament, "திருவருட்சாதன")

        if lang == "ta":
            resolved_query = f"{pname} குடும்பத்தினரின் {s_name_ta} பதிவுகளை காட்டு{card_str_ta}."
        else:
            resolved_query = f"{s_name_en} records of {pname}'s family{card_str}."

    elif action == "MEMBER_COUNT":
        scope = "FAMILY_MEMBER_COUNT"
        c_intent = "FAMILY_SEARCH"
        sem_intent = "FAMILY_MEMBER_COUNT"
        db_op = "COUNT_FAMILY_MEMBERS"
        req_info = "MEMBER_COUNT"
        if lang == "ta":
            resolved_query = f"{pname} குடும்பத்தில் எத்தனை நபர்கள் உள்ளார்கள்{card_str_ta}?"
        else:
            resolved_query = f"How many people are in {pname}'s family{card_str}?"

    elif action == "CONTACT_DETAILS":
        scope = "FAMILY_DETAILS"
        c_intent = "FAMILY_SEARCH"
        sem_intent = "FAMILY_DETAILS"
        db_op = "RETRIEVE_FAMILY_BY_ID"
        req_info = "MEMBER_CONTACT"
        if lang == "ta":
            resolved_query = f"{pname} குடும்ப தொடர்பு விவரங்களை காட்டு{card_str_ta}."
        else:
            resolved_query = f"Show contact details of {pname}'s family{card_str}."

    intent_info = {
        "original_query": original_query,
        "resolved_query": resolved_query,
        "corrected_query": resolved_query,
        "intent": c_intent,
        "semantic_intent": sem_intent,
        "scope": scope,
        "target_scope": "FAMILY_MEMBERS",
        "database_operation": db_op,
        "requested_information": req_info,
        "person_name": pname,
        "family_card": card,
        "family_id": fid,
        "parish": parish,
        "sacrament": sacrament,
        "speed_tier": "FAST",
        "detected_language": lang,
        "final_response_language": "ta" if lang == "ta" else "en",
        "inherited_from_previous_turn": True,
    }

    return resolved_query, intent_info
