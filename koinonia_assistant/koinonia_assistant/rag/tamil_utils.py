import re
from typing import Dict, Any, List, Optional, Tuple

# ─── 1. Unicode Script & Language Detection (Section 28) ─────────────────────
TAMIL_UNICODE_RE = re.compile(r"[\u0B80-\u0BFF]")
MALAYALAM_UNICODE_RE = re.compile(r"[\u0D00-\u0D7F]")
TELUGU_UNICODE_RE = re.compile(r"[\u0C00-\u0C7F]")
KANNADA_UNICODE_RE = re.compile(r"[\u0C80-\u0CFF]")
HINDI_UNICODE_RE = re.compile(r"[\u0900-\u097F]")
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]{2,}")


def is_tamil(text: str) -> bool:
    """Returns True if the text contains Tamil Unicode characters (U+0B80–U+0BFF)."""
    if not text:
        return False
    return bool(TAMIL_UNICODE_RE.search(text))


def detect_query_language(text: str) -> str:
    """
    Detects the language of the input query directly from script characters.
    Never transliterates Tamil into Tanglish.
    Returns: 'ta', 'ta-en' (when mixed Tamil-English), 'ml', 'te', 'kn', 'hi', or 'en'.
    """
    if not text:
        return "en"
    if TAMIL_UNICODE_RE.search(text):
        return "ta"
    if MALAYALAM_UNICODE_RE.search(text):
        return "ml"
    if TELUGU_UNICODE_RE.search(text):
        return "te"
    if KANNADA_UNICODE_RE.search(text):
        return "kn"
    if HINDI_UNICODE_RE.search(text):
        return "hi"
    return "en"


# ─── 2. Domain-Specific Catholic Tamil Terminology Dictionary (Sections 30, 31, 41) ───
CATHOLIC_TAMIL_TERMINOLOGY_DB: List[Dict[str, Any]] = [
    {
        "term_id": "SAC_BAPTISM",
        "language": "ta",
        "category": "SACRAMENT",
        "code": "BAPTISM",
        "metric_key": "baptism",
        "canonical_term": "திருமுழுக்கு",
        "secondary_term": "ஞானஸ்நானம்",
        "english_term": "Baptism",
        "aliases": [
            "திருமுழுக்கு",
            "திருமுழுக்குச் சடங்கு",
            "திருமுழுக்கு சடங்கு",
            "ஞானஸ்நானம்",
            "ஞானஸ்நான",
            "ஞானस्नाనం",
            "baptism",
            "baptisms",
            "baptism record",
            "baptism status",
        ],
        "active": True,
    },
    {
        "term_id": "SAC_FHC",
        "language": "ta",
        "category": "SACRAMENT",
        "code": "FIRST_HOLY_COMMUNION",
        "metric_key": "communion",
        "canonical_term": "முதல் நற்கருணை",
        "secondary_term": "முதல் திருவிருந்து",
        "english_term": "First Holy Communion",
        "aliases": [
            "முதல் நற்கருணை",
            "முதல் நற்கருணை அருட்சாதனம்",
            "முதல் திருவிருந்து",
            "நற்கருணை",
            "புதுநன்மை",
            "first holy communion",
            "holy communion",
            "communion",
            "fhc",
        ],
        "active": True,
    },
    {
        "term_id": "SAC_CONFIRMATION",
        "language": "ta",
        "category": "SACRAMENT",
        "code": "CONFIRMATION",
        "metric_key": "confirmation",
        "canonical_term": "உறுதிப்பூசுதல்",
        "secondary_term": "உறுதிப்படுத்துதல்",
        "english_term": "Confirmation",
        "aliases": [
            "உறுதிப்பூசுதல்",
            "உறுதிபூசுதல்",
            "உறுதிப்படுத்துதல்",
            "உறுதிப்படுத்தல்",
            "confirmation",
            "confirmations",
        ],
        "active": True,
    },
    {
        "term_id": "SAC_MARRIAGE",
        "language": "ta",
        "category": "SACRAMENT",
        "code": "MARRIAGE",
        "metric_key": "marriage",
        "canonical_term": "திருமணம்",
        "secondary_term": "திருமண அருட்சாதனம்",
        "english_term": "Marriage",
        "aliases": [
            "திருமணம்",
            "திருமண அருட்சாதனம்",
            "திருமண",
            "விவாகம்",
            "marriage",
            "marriages",
            "wedding",
        ],
        "active": True,
    },
    {
        "term_id": "GEN_SACRAMENT",
        "language": "ta",
        "category": "CONCEPT",
        "code": "SACRAMENT",
        "canonical_term": "அருட்சாதனம்",
        "english_term": "Sacrament",
        "aliases": ["அருட்சாதனம்", "திருவருட்சாதனம்", "திருவருட்சாதனங்கள்", "அருட்சாதனங்கள்"],
        "active": True,
    },
    {
        "term_id": "ORG_PARISH",
        "language": "ta",
        "category": "ORGANIZATION",
        "code": "PARISH",
        "canonical_term": "பங்கு",
        "secondary_term": "பங்குத்தளம்",
        "english_term": "Parish",
        "aliases": ["பங்குத்தளம்", "பங்கு", "பங்கில்", "பங்கின்", "கிளைப்பங்கு"],
        "active": True,
    },
    {
        "term_id": "ORG_PRIEST",
        "language": "ta",
        "category": "ORGANIZATION",
        "code": "PARISH_PRIEST",
        "canonical_term": "பங்குத்தந்தை",
        "english_term": "Parish Priest",
        "aliases": ["பங்குத்தந்தை", "பங்குத் தந்தை"],
        "active": True,
    },
    {
        "term_id": "ORG_DIOCESE",
        "language": "ta",
        "category": "ORGANIZATION",
        "code": "DIOCESE",
        "canonical_term": "மறைமாவட்டம்",
        "english_term": "Diocese",
        "aliases": ["மறைமாவட்டம்", "மறைமாவட்ட", "மறைமாவட்டத்தில்"],
        "active": True,
    },
    {
        "term_id": "ENT_FAMILY",
        "language": "ta",
        "category": "ENTITY",
        "code": "FAMILY",
        "canonical_term": "குடும்பம்",
        "english_term": "Family",
        "aliases": ["குடும்பம்", "குடும்பத்தின்", "குடும்ப", "குடும்பங்கள்"],
        "active": True,
    },
    {
        "term_id": "ENT_FAMILY_CARD",
        "language": "ta",
        "category": "ENTITY",
        "code": "FAMILY_CARD",
        "canonical_term": "குடும்ப அட்டை",
        "english_term": "Family Card",
        "aliases": ["குடும்ப அட்டை", "குடும்பப் பதிவு எண்"],
        "active": True,
    },
    {
        "term_id": "ENT_MEMBER",
        "language": "ta",
        "category": "ENTITY",
        "code": "MEMBER",
        "canonical_term": "உறுப்பினர்",
        "english_term": "Member",
        "aliases": ["உறுப்பினர்", "உறுப்பினர்கள்", "உறுப்பினர்களை", "பங்கு மக்கள்", "நபர்கள்"],
        "active": True,
    },
]


# ─── 3. Tamil Person Name Dictionary & Optional Search Matching Helper (Sections 32, 33) ───
TAMIL_NAME_DICTIONARY = {
    "அந்தோணி ராஜ்": "Antony Raj",
    "அந்தோனி ராஜ்": "Antony Raj",
    "அந்தோனிராஜ்": "Antony Raj",
    "அந்தோணிராஜ்": "Antony Raj",
    "அந்தோணி செல்வம்": "Antony Selvam",
    "அந்தோனி செல்வம்": "Antony Selvam",
    "அந்தோணி செல்வன்": "Antony Selvan",
    "அந்தோனி செல்வன்": "Antony Selvan",
    "ஆரோக்கியதாஸ்": "Arokiadass",
    "சகாயராஜ்": "Sahayaraj",
    "குழந்தைராஜ்": "Kulandairaj",
    "அற்புதராஜ்": "Arputharaj",
    "அந்தோணி": "Antony",
    "அந்தோனி": "Antony",
    "ஆண்டனி": "Antony",
    "ராஜ்": "Raj",
    "ராஜா": "Raja",
    "செல்வம்": "Selvam",
    "செல்வன்": "Selvan",
    "ஆல்பர்ட்": "Albert",
    "ரட்சகர்": "Ratchagar",
    "மரிய": "Mary",
    "மேரி": "Mary",
    "மைக்கேல்": "Michael",
    "ஜோசப்": "Joseph",
    "பீட்டர்": "Peter",
    "தாமஸ்": "Thomas",
    "பிரான்சிஸ்": "Francis",
    "இஞ்ஞாசி": "Ignatius",
    "ராபர்ட்": "Robert",
    "ஆக்னஸ்": "Agnes",
}

TAMIL_INITIALS_MAP = {
    "எஸ்": "S", "ஏ": "A", "பி": "B", "சி": "C", "டி": "D",
    "இ": "E", "எப்": "F", "ஜி": "G", "ஐ": "I", "ஜே": "J",
    "கே": "K", "எல்": "L", "எம்": "M", "என்": "N", "ஆர்": "R", "வி": "V",
}

TAMIL_VOWELS = {
    'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ee', 'உ': 'u', 'ஊ': 'oo',
    'எ': 'e', 'ஏ': 'ae', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oa', 'ஔ': 'au'
}

TAMIL_CONSONANTS = {
    'க': 'k', 'ங': 'ng', 'ச': 's', 'ஞ': 'nj', 'ட': 't', 'ண': 'n',
    'த': 'th', 'ந': 'n', 'ப': 'p', 'ம': 'm', 'ய': 'y', 'ர': 'r',
    'ல': 'l', 'வ': 'v', 'ழ': 'zh', 'ள': 'l', 'ற': 'r', 'ன': 'n',
    'ஜ': 'j', 'ஷ': 'sh', 'ஸ': 's', 'ஹ': 'h', 'க்ஷ': 'ksh'
}

TAMIL_VOWEL_SIGNS = {
    'ா': 'aa', 'ி': 'i', 'ீ': 'ee', 'ு': 'u', 'ூ': 'oo',
    'ெ': 'e', 'ே': 'ae', 'ை': 'ai', 'ொ': 'o', 'ோ': 'oa', 'ௌ': 'au'
}

VIRAMA = '்'


def transliterate_tamil_word(word: str) -> str:
    """Used ONLY for person-name lookup matching against Latin DB names. NEVER used on user queries."""
    if not is_tamil(word):
        return word
    if word in TAMIL_INITIALS_MAP:
        return TAMIL_INITIALS_MAP[word]
    out = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch in TAMIL_VOWELS:
            out.append(TAMIL_VOWELS[ch])
            i += 1
            continue
        if ch in TAMIL_CONSONANTS:
            base_cons = TAMIL_CONSONANTS[ch]
            if i + 1 < n:
                next_ch = word[i + 1]
                if next_ch == VIRAMA:
                    out.append(base_cons)
                    i += 2
                    continue
                elif next_ch in TAMIL_VOWEL_SIGNS:
                    out.append(base_cons + TAMIL_VOWEL_SIGNS[next_ch])
                    i += 2
                    continue
            out.append(base_cons + 'a')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out).capitalize()


def transliterate_tamil_name(name_str: str) -> str:
    """
    Converts a Tamil person name into its English database matching equivalent
    ONLY for searching tabMember when member names are stored in Latin script.
    Preserves original_name separately.
    """
    if not name_str or not is_tamil(name_str):
        return name_str
    clean = re.sub(r"(?:யின்|வின்|ன்|க்கு|உடைய|அவர்களின்)$", "", name_str.strip()).strip()
    for tam, eng in sorted(TAMIL_NAME_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True):
        if tam in clean:
            clean = clean.replace(tam, f" {eng} ")
    tokens = clean.split()
    out = []
    for tok in tokens:
        tok_stripped = re.sub(r"(?:யின்|வின்|ன்|க்கு|உடைய)$", "", tok)
        if is_tamil(tok_stripped):
            if tok_stripped in TAMIL_INITIALS_MAP:
                out.append(TAMIL_INITIALS_MAP[tok_stripped])
            elif tok_stripped in TAMIL_NAME_DICTIONARY:
                out.append(TAMIL_NAME_DICTIONARY[tok_stripped])
            else:
                out.append(transliterate_tamil_word(tok_stripped))
        else:
            out.append(tok_stripped)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def normalize_tamil_person_entity(raw_name: str) -> Dict[str, Optional[str]]:
    """
    Returns separate representations for a person name (Section 33):
    - original_name
    - normalized_name
    - transliterated_name
    """
    if not raw_name:
        return {"original_name": None, "normalized_name": None, "transliterated_name": None}
    orig = raw_name.strip()
    norm_ta = re.sub(r"(?:யின்|வின்|ன்|க்கு|உடைய|அவர்களின்|-க்கு)$", "", orig).strip()
    trans = transliterate_tamil_name(norm_ta) if is_tamil(norm_ta) else norm_ta
    return {
        "original_name": orig,
        "normalized_name": norm_ta,
        "transliterated_name": trans,
    }


# ─── 4. Direct Tamil & Multilingual Structured Intent & Entity Extractor (Sections 29, 34, 35) ───
def extract_tamil_structured_intent(original_query: str) -> Dict[str, Any]:
    """
    Directly understands Tamil and mixed Tamil-English queries WITHOUT converting
    the user's sentence into Tanglish.
    Extracts:
    - original_query (immutable Tamil text)
    - detected_language ('ta' or 'en' etc.)
    - intent (HISTORICAL_ANALYSIS, TREND_ANALYSIS, FORECAST, STATISTICAL_ANALYSIS, COMPARISON, SACRAMENT_SEARCH, FAMILY_SEARCH, MEMBER_SEARCH, LIST, COUNT)
    - sacrament (FIRST_HOLY_COMMUNION, BAPTISM, CONFIRMATION, MARRIAGE)
    - metrics (list of metric keys: ['communion'], ['baptism'], etc.)
    - canonical_terms (dict of matched canonical Tamil terms)
    - period (e.g. 'LAST_10_YEARS')
    - years_back (int)
    - forecast_horizon (int)
    - grouping ('YEAR' | None)
    - family_card (e.g. 'YLG/001')
    - person_entity ({original_name, normalized_name, transliterated_name})
    """
    q = (original_query or "").strip()
    lang = detect_query_language(q)

    # 1. Match Canonical Catholic Tamil Terminology
    matched_sacraments: List[Dict[str, Any]] = []
    canonical_terms: Dict[str, str] = {}

    # Sort sacrament aliases by length descending so 'முதல் நற்கருணை' matches before 'நற்கருணை'
    for entry in CATHOLIC_TAMIL_TERMINOLOGY_DB:
        for alias in sorted(entry["aliases"], key=len, reverse=True):
            if alias.lower() in q.lower():
                canonical_terms[entry["code"]] = entry["canonical_term"]
                if entry["category"] == "SACRAMENT" and entry not in matched_sacraments:
                    matched_sacraments.append(entry)
                break

    metrics = [m["metric_key"] for m in matched_sacraments] if matched_sacraments else []
    primary_sacrament_code = matched_sacraments[0]["code"] if matched_sacraments else None

    # 2. Extract Time Period / Horizon from Tamil & English numbers
    years_back = 10
    m_hist = re.search(r"(?:கடந்த|முந்தைய|last|past)\s+(\d+)\s*(?:ஆண்டுகளில்|ஆண்டுகள்|வருடங்களில்|வருடங்கள்|years?)", q, re.IGNORECASE)
    if m_hist:
        years_back = max(1, min(50, int(m_hist.group(1))))
    period_code = f"LAST_{years_back}_YEARS"

    forecast_horizon = 0
    has_forecast = bool(
        re.search(r"(?:அடுத்த|வரும்|எதிர்கால|கணிப்பு|கணிப்பை|முன்கணிப்பு|எப்படி\s+இருக்கும்|forecast|predict|projection|next\s+\d+\s+years)", q, re.IGNORECASE)
    )
    if has_forecast:
        m_f = re.search(r"(?:அடுத்த|வரும்|next|upcoming)\s+(\d+)\s*(?:ஆண்டுகளுக்கு|ஆண்டுகளில்|ஆண்டுகள்|வருடங்கள்|years?)", q, re.IGNORECASE)
        forecast_horizon = int(m_f.group(1)) if m_f else years_back

    # 3. Extract Grouping (Year-by-Year)
    has_yearly_grouping = bool(
        re.search(
            r"(?:ஒவ்வொரு\s+ஆண்டும்|ஆண்டுதோறும்|ஆண்டு\s*வாரியாக|வருடந்தோறும்|ஒவ்வொரு\s+வருடமும்|each\s+year|every\s+year|year\s+by\s+year|yearly|annual)",
            q,
            re.IGNORECASE,
        )
    )
    grouping = "YEAR" if (has_yearly_grouping or m_hist or has_forecast) else None

    # 4. Extract Family Card / Member ID
    m_card = re.search(r"\b([A-Z]{2,5}/\d{1,5})\b", q, re.IGNORECASE)
    family_card = m_card.group(1).upper() if m_card else None

    # 5. Determine Canonical Intent
    has_trend = bool(re.search(r"(?:மாற்றம்|மாற்றத்தை|போக்கு|வளர்ச்சி|குறைவு|அதிகரிப்பு|trend|changed|growth)", q, re.IGNORECASE))
    has_stats = bool(re.search(r"(?:புள்ளிவிவரம்|புள்ளிவிவரங்கள்|சராசரி|statistics|statistical|summary)", q, re.IGNORECASE))
    has_comparison = bool(re.search(r"(?:ஒப்பிடு|ஒப்பீடு|compare|comparison)", q, re.IGNORECASE)) or len(metrics) >= 2
    has_status = bool(re.search(r"(?:நிலை\s+என்ன|நிலை|status|விவரம்|விவரங்கள்|details)", q, re.IGNORECASE))
    has_family = bool(re.search(r"(?:குடும்பத்தின்|குடும்ப|உறுப்பினர்களை|family)", q, re.IGNORECASE))
    has_count = bool(re.search(r"(?:எண்ணிக்கை|எண்ணிக்கையைத்|எண்ணிக்கையை|எத்தனை|மொத்தம்|how\s+many|count)", q, re.IGNORECASE))
    has_list = bool(re.search(r"(?:காட்டு|பட்டியலிடுக|பட்டியல்|தரவும்|கொடு|list|show)", q, re.IGNORECASE))

    intent = "GENERAL_DATABASE_QUERY"
    if has_comparison and (m_hist or has_yearly_grouping or has_forecast or has_trend or has_stats):
        intent = "COMPARISON"
    elif has_forecast:
        intent = "FORECAST"
    elif has_trend and (m_hist or has_yearly_grouping):
        intent = "TREND_ANALYSIS"
    elif has_stats and (m_hist or has_yearly_grouping):
        intent = "STATISTICAL_ANALYSIS"
    elif (m_hist or has_yearly_grouping) and (has_count or has_list or metrics):
        intent = "HISTORICAL_ANALYSIS"
    elif family_card and has_family:
        intent = "FAMILY_SEARCH"
    elif has_status and metrics:
        intent = "SACRAMENT_SEARCH"
    elif has_family and not m_hist:
        intent = "FAMILY_SEARCH"
    elif has_count and not m_hist and not has_yearly_grouping:
        intent = "COUNT"
    elif has_list and not m_hist and not has_yearly_grouping:
        intent = "LIST"

    # 6. Extract Person Name if SACRAMENT_SEARCH / FAMILY_SEARCH / MEMBER_SEARCH
    person_entity = {"original_name": None, "normalized_name": None, "transliterated_name": None}
    if intent in ("SACRAMENT_SEARCH", "FAMILY_SEARCH", "MEMBER_SEARCH") and not family_card:
        # Look for known Tamil names or '<name>-க்கு' / '<name>ன்' / '<name> குடும்பத்தின்'
        for tam_name in sorted(TAMIL_NAME_DICTIONARY.keys(), key=len, reverse=True):
            if tam_name in q:
                person_entity = normalize_tamil_person_entity(tam_name)
                break
        if not person_entity["original_name"]:
            m_p = re.search(r"^([^\s]+(?:\s+[^\s]+)?)(?:யின்|வின்|ன்|-க்கு|க்கு)\s+", q)
            if m_p:
                person_entity = normalize_tamil_person_entity(m_p.group(1))
            else:
                m_eng_p = re.search(r"([A-Za-z]+(?:\s+[A-Za-z\.]+)?)(?:-க்கு|\s+baptism|\s+status|\s+family)", q, re.IGNORECASE)
                if m_eng_p:
                    person_entity = normalize_tamil_person_entity(m_eng_p.group(1))

    # Build structured representation (NEVER Tanglish)
    structured_repr = (
        f"INTENT={intent} | LANGUAGE={lang} | SACRAMENT={primary_sacrament_code or 'NONE'} "
        f"| PERIOD={period_code} | GROUPING={grouping or 'NONE'} "
        f"| FAMILY_CARD={family_card or 'NONE'} | PERSON={person_entity.get('original_name') or 'NONE'}"
    )

    return {
        "original_query": q,
        "detected_language": lang,
        "normalized_query": structured_repr,
        "intent_query": structured_repr,
        "entity_query": structured_repr,
        "intent": intent,
        "sacrament": primary_sacrament_code,
        "metrics": metrics or ["baptism"],
        "canonical_terms": canonical_terms,
        "period": period_code,
        "years_back": years_back,
        "forecast_horizon": forecast_horizon,
        "grouping": grouping,
        "family_card": family_card,
        "person_entity": person_entity,
        "final_response_language": "ta" if lang == "ta" else "en",
    }


def normalize_tamil_query(query_text: str) -> str:
    """
    DEPRECATED Tanglish converter replaced in accordance with Sections 24–27 & 48:
    Returns the original Tamil query untouched so no caller can accidentally
    overwrite the user's Tamil query with Tanglish.
    """
    return query_text
