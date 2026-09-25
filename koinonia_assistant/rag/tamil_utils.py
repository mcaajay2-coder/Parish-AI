import datetime
import re
from typing import Dict, Any, List, Optional, Tuple

# ─── 1. Unicode Script & Language Detection (Section 28) ─────────────────────
TAMIL_UNICODE_RE = re.compile(r"[\u0B80-\u0BFF]")
MALAYALAM_UNICODE_RE = re.compile(r"[\u0D00-\u0D7F]")
TELUGU_UNICODE_RE = re.compile(r"[\u0C00-\u0C7F]")
KANNADA_UNICODE_RE = re.compile(r"[\u0C80-\u0CFF]")
HINDI_UNICODE_RE = re.compile(r"[\u0900-\u097F]")
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]{2,}")

TANGLISH_MARKERS = {
    "kudumbam", "kudumbathil", "urupinargal", "urupinar", "ethanai", "eththanai",
    "evlo", "evalo", "irukanga", "irukkanga", "ullargal", "irukku", "irukkum",
    "nadanthuchu", "nadanthathu", "aachu", "maariyullathu", "eppadi", "enna",
    "kudu", "sollu", "kaatu", "kaatunga", "avargalin", "avargal", "yaar",
    "puthunanmai", "pothunanmai", "gnanasnanam", "thirumulukku", "thirumanam",
    "urudhipoosuthal", "narkarunai", "pangu", "anbiyam",
}


def is_tamil(text: str) -> bool:
    """Returns True if the text contains Tamil Unicode characters (U+0B80–U+0BFF)."""
    if not text:
        return False
    return bool(TAMIL_UNICODE_RE.search(text))


def is_tanglish(text: str) -> bool:
    """Returns True if Latin-script text contains recognizable Tanglish markers."""
    if not text or is_tamil(text):
        return False
    words = {w.lower().strip(".,?!'\":;-") for w in text.split()}
    return bool(words & TANGLISH_MARKERS)


def detect_query_language(text: str) -> str:
    """
    Detects the language of the input query directly from script characters & vocabulary.
    Never transliterates Tamil into Tanglish.
    Returns: 'ta', 'ml', 'te', 'kn', 'hi', 'tanglish', or 'en'.
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
    if is_tanglish(text):
        return "tanglish"
    return "en"


# ─── 2. Domain-Specific Catholic Tamil Terminology Dictionary (Section 12) ───
CATHOLIC_TAMIL_TERMINOLOGY_DB: List[Dict[str, Any]] = [
    {
        "term_id": "SAC_BAPTISM",
        "language": "ta",
        "category": "SACRAMENT",
        "code": "BAPTISM",
        "metric_key": "baptism",
        "canonical_term": "ஞானஸ்நானம்",
        "secondary_term": "திருமுழுக்கு",
        "english_term": "Baptism",
        "aliases": [
            "திருமுழுக்கு",
            "திருமுழுக்குச் சடங்கு",
            "திருமுழுக்கு சடங்கு",
            "ஞானஸ்நானம்ச் சடங்குகள்",
            "ஞானஸ்நானச் சடங்குகள்",
            "ஞானஸ்நானச் சடங்கு",
            "ஞானஸ்நான சடங்கு",
            "ஞானஸ்நானங்கள்",
            "ஞானஸ்நானம்",
            "ஞானஸ்நான",
            "ஞானस्नाనం",
            "gnanasnanam",
            "thirumulukku",
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
            "பொதுநன்மை",
            "puthunanmai",
            "pothunanmai",
            "narkarunai",
            "first holy communion",
            "first communion",
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
            "urudhipoosuthal",
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
            "திருமணங்கள்",
            "திருமண",
            "விவாகம்",
            "thirumanam",
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
        "canonical_term": "திருவருட்சாதனம்",
        "english_term": "Sacrament",
        "aliases": [
            "அருட்சாதனம்",
            "திருவருட்சாதனம்",
            "திருவருட்சாதனங்கள்",
            "அருட்சாதனங்கள்",
            "சக்கரமெண்ட்",
            "sacrament",
            "sacraments",
        ],
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
        "aliases": ["பங்குத்தளம்", "பங்கு", "பங்கில்", "பங்கின்", "கிளைப்பங்கு", "pangu"],
        "active": True,
    },
    {
        "term_id": "ORG_VICARIATE",
        "language": "ta",
        "category": "ORGANIZATION",
        "code": "VICARIATE",
        "canonical_term": "வட்டாரம்",
        "english_term": "Vicariate",
        "aliases": ["வட்டாரம்", "மறைவட்டம்", "vicariate"],
        "active": True,
    },
    {
        "term_id": "ORG_ANBIYAM",
        "language": "ta",
        "category": "ORGANIZATION",
        "code": "ANBIYAM",
        "canonical_term": "அன்பியம்",
        "english_term": "Anbiyam",
        "aliases": ["அன்பியம்", "அன்பிய", "anbiyam", "bcc"],
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
        "aliases": ["குடும்பம்", "குடும்பத்தில்", "குடும்பத்தின்", "குடும்ப", "குடும்பங்கள்", "kudumbam", "kudumbathil", "family"],
        "active": True,
    },
    {
        "term_id": "ENT_FAMILY_CARD",
        "language": "ta",
        "category": "ENTITY",
        "code": "FAMILY_CARD",
        "canonical_term": "குடும்ப அட்டை",
        "english_term": "Family Card",
        "aliases": ["குடும்ப அட்டை", "குடும்பப் பதிவு எண்", "family card"],
        "active": True,
    },
    {
        "term_id": "ENT_MEMBER",
        "language": "ta",
        "category": "ENTITY",
        "code": "MEMBER",
        "canonical_term": "உறுப்பினர்",
        "english_term": "Member",
        "aliases": [
            "குடும்ப உறுப்பினர்",
            "குடும்ப உறுப்பினர்கள்",
            "உறுப்பினர்",
            "உறுப்பினர்கள்",
            "உறுப்பினர்களை",
            "பங்கு மக்கள்",
            "நபர்கள்",
            "பேர்",
            "அங்கத்தினர்கள்",
            "urupinargal",
            "members",
            "member",
            "people",
        ],
        "active": True,
    },
]


# ─── 3. Tamil Person Name Dictionary & Entity Extraction Helpers ─────────────
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
    "ரோஸ்லின்": "Roselin",
    "ரோசலின்": "Roselin",
    "ரோஸ்லைன்": "Roseline",
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

# Non-name words (English + Tanglish) that must NEVER be extracted as part of a person entity
MULTILINGUAL_NON_NAME_WORDS = {
    "family", "families", "member", "members", "people", "person", "parishioner", "parishioners",
    "details", "detail", "info", "information", "record", "records", "status", "card", "id",
    "baptism", "baptisms", "baptized", "baptised", "communion", "fhc", "eucharist", "confirmation",
    "confirmations", "confirmed", "marriage", "marriages", "married", "wedding", "matrimony",
    "death", "deceased", "burial", "sacrament", "sacraments", "sacrement", "sacrements",
    "count", "total", "number", "how", "many", "who", "what", "where", "when", "why", "which",
    "is", "are", "was", "were", "in", "of", "for", "about", "the", "a", "an", "to", "from", "with",
    "show", "give", "get", "find", "list", "view", "display", "tell", "me", "all", "any", "some",
    "phone", "mobile", "contact", "address", "place", "street", "parish", "yelagiri", "diocese",
    "last", "past", "previous", "next", "coming", "upcoming", "future", "year", "years", "each",
    "every", "yearly", "annual", "trend", "change", "changed", "forecast", "predict", "statistics",
    "chart", "graph", "plot", "visualize", "visualise", "compare", "comparison",
    # Tanglish particles & verbs
    "oda", "ku", "kku", "la", "le", "il", "evlo", "evalo", "ethanai", "eththanai", "irukanga",
    "irukkanga", "ullargal", "irukku", "irukkum", "nadanthuchu", "nadanthathu", "aachu", "eppadi",
    "enna", "kudu", "kudunga", "sollu", "sollunga", "kaatu", "kaatunga", "avargalin", "avargal",
    "yaar", "yar", "per", "peru", "nambargal", "nabargal", "urupinargal", "urupinar", "kudumbam",
    "kudumbathil", "pangu", "pangil", "gnanasnanam", "thirumulukku", "thirumanam",
    "urudhipoosuthal", "narkarunai", "puthunanmai", "pothunanmai",
    # Pronoun & anaphoric tokens (MUST NEVER be extracted as person names)
    "them", "they", "their", "theirs", "him", "her", "his", "he", "she", "it", "this", "these", "those",
    "avanga", "avangala", "avangaloda", "avangaluku", "avangalku", "avaru", "avar", "ivanga", "ivangala", "ivangaloda",
    "pannu", "sol",
}


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
    clean = re.sub(r"(?:யின்|வின்|ன்|க்கு|உடைய|அவர்களின்|அவர்கள்)$", "", name_str.strip()).strip()
    for tam, eng in sorted(TAMIL_NAME_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True):
        if tam in clean:
            clean = clean.replace(tam, f" {eng} ")
    tokens = clean.split()
    out = []
    for tok in tokens:
        tok_stripped = re.sub(r"(?:யின்|வின்|ன்|க்கு|உடைய|அவர்களின்|அவர்கள்)$", "", tok)
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
    Returns separate representations for a person name (Section 15):
    - original_name
    - normalized_name
    - transliterated_name
    """
    if not raw_name:
        return {"original_name": None, "normalized_name": None, "transliterated_name": None}
    orig = raw_name.strip()
    norm_ta = re.sub(r"(?:யின்|வின்|ன்|க்கு|உடைய|அவர்களின்|அவர்கள்|-க்கு|-ல்|-இல்)$", "", orig).strip()
    trans = transliterate_tamil_name(norm_ta) if is_tamil(norm_ta) else norm_ta
    return {
        "original_name": orig,
        "normalized_name": norm_ta,
        "transliterated_name": trans,
    }


def extract_person_entity_from_multilingual_query(query_text: str) -> Dict[str, Optional[str]]:
    """
    Extracts person entity from Tamil, Tanglish, Mixed Tamil-English, or English queries
    WITHOUT losing initials or modifying the name (Section 15).
    Examples:
      - "Antony Selvan குடும்பம் எத்தனை நபர்கள் உள்ளார்கள்" -> "Antony Selvan"
      - "Antony Selvan குடும்பம் அவர்கள் மொத்தம் எத்தனை நபர்கள் உள்ளார்கள்" -> "Antony Selvan"
      - "Antony Selvan P குடும்பத்தில் யார் யார் இருக்கிறார்கள்?" -> "Antony Selvan P"
      - "Antony Selvan oda family la evlo members irukanga?" -> "Antony Selvan"
      - "Antony Selvan-க்கு baptism status என்ன?" -> "Antony Selvan"
      - "அந்தோணி செல்வன் குடும்பத்தில் எத்தனை பேர்?" -> "Antony Selvan"
    """
    if not query_text:
        return {"original_name": None, "normalized_name": None, "transliterated_name": None}

    q_no_cards = re.sub(r"\b[A-Z]{2,5}/\d{1,5}\b", "", query_text, flags=re.IGNORECASE)
    q_no_cards = re.sub(r"\s*\((?:Member\s*ID|Family\s*ID|Family|ID|Card)[:\s0-9A-Za-z,\s\-/]+\)", "", q_no_cards, flags=re.IGNORECASE)

    # 1. Check known Tamil dictionary names first
    for tam_name in sorted(TAMIL_NAME_DICTIONARY.keys(), key=len, reverse=True):
        if tam_name in q_no_cards:
            return normalize_tamil_person_entity(tam_name)

    # 2. Extract contiguous Latin name tokens (handles Mixed Tamil-English & Tanglish & English)
    # Strip attached Tamil suffixes like '-க்கு', '-ல்', 'க்கு', 'யின்' from Latin tokens
    latin_tokens = []
    raw_tokens = re.findall(r"[A-Za-z]+(?:\.[A-Za-z]+)?", q_no_cards)
    for idx, tok in enumerate(raw_tokens):
        t_clean = tok.strip(".")
        t_low = t_clean.lower()
        if not t_low:
            continue
        # Allow single-letter surname initial (e.g. 'P', 'A', 'S') if we already have at least 1 name word
        if len(t_clean) == 1 and latin_tokens:
            latin_tokens.append(t_clean.upper())
        elif len(t_clean) >= 2 and t_low not in MULTILINGUAL_NON_NAME_WORDS:
            latin_tokens.append(t_clean)
        elif latin_tokens:
            # Stop once we hit an intent/stopword after collecting the person's name tokens
            break

    if latin_tokens:
        extracted = " ".join(latin_tokens)
        return normalize_tamil_person_entity(extracted)

    # 3. Pure Tamil query prefix before 'குடும்ப' / 'அவர்களின்' / 'அவர்கள்' / 'என்பவரின்'
    # Guard against time/aggregate queries and pronoun follow-ups starting with pronouns or time words
    if not re.match(r"^\s*(?:கடந்த|இந்த|அடுத்த|முந்தைய|போன|நடப்பு|ஒவ்வொரு|மொத்தம்|எத்தனை|நமது|எங்கள்|பங்கில்|பங்கு|யார்|என்ன|எப்படி|எவ்வாறு|அவர்கள்|அவர்களை|அவர்களின்|அவர்களுடைய|அவர்|அவரை|அவரின்|இவர்|இவரை|இவர்கள்|இவர்களை|இவர்களின்)\b", q_no_cards):
        m_ta_prefix = re.search(
            r"^\s*([\u0B80-\u0BFF\s]{3,40}?)\s+(?:குடும்பம்|குடும்பத்தில்|குடும்பத்தின்|குடும்ப|அவர்களின்|அவர்கள்|என்பவரின்|என்பவர்)",
            q_no_cards,
        )
        if m_ta_prefix:
            cand_ta = m_ta_prefix.group(1).strip()
            ta_stopwords = {
                "கடந்த", "அடுத்த", "இந்த", "முந்தைய", "போன", "நடப்பு", "ஆண்டு", "ஆண்டுகள்",
                "ஆண்டுகளில்", "ஆண்டுகளின்", "வருடம்", "வருடங்கள்", "வருடங்களில்", "மாதம்",
                "மொத்தம்", "எத்தனை", "நமது", "எங்கள்", "பங்கில்", "பங்கு", "ஒவ்வொரு", "அனைத்து",
                "யார்", "என்ன", "எப்படி", "எவ்வாறு",
                "அவர்கள்", "அவர்களை", "அவர்களின்", "அவர்களுடைய", "அவர்", "அவரை", "அவரின்", "அவருடைய",
                "இவர்", "இவரை", "இவர்கள்", "இவர்களை", "இவர்களின்", "இவருடைய"
            }
            cand_words = [w for w in cand_ta.split() if w not in ta_stopwords and not w.isdigit()]
            if cand_words:
                return normalize_tamil_person_entity(" ".join(cand_words))

    return {"original_name": None, "normalized_name": None, "transliterated_name": None}


# ─── 4. Direct Tamil & Multilingual Structured Intent & Entity Extractor (Sections 1–21) ───
def extract_tamil_structured_intent(original_query: str) -> Dict[str, Any]:
    """
    Directly understands Tamil, Mixed Tamil-English, and Tanglish queries WITHOUT
    converting the user's sentence into a blind English/Tanglish translation.
    Returns the complete semantic representation (Section 3 & Section 21).
    """
    q = (original_query or "").strip()
    q_low = q.lower()
    lang = detect_query_language(q)
    cur_year = datetime.date.today().year

    # 1. Match Canonical Catholic Tamil & Multilingual Terminology
    matched_sacraments: List[Dict[str, Any]] = []
    canonical_terms: Dict[str, str] = {}

    for entry in CATHOLIC_TAMIL_TERMINOLOGY_DB:
        for alias in sorted(entry["aliases"], key=len, reverse=True):
            if alias.lower() in q_low:
                canonical_terms[entry["code"]] = entry["canonical_term"]
                if entry["category"] == "SACRAMENT" and entry not in matched_sacraments:
                    matched_sacraments.append(entry)
                break

    metrics = [m["metric_key"] for m in matched_sacraments] if matched_sacraments else []
    primary_sacrament_code = matched_sacraments[0]["code"] if matched_sacraments else None

    # 2. Extract Time Expressions (Section 11: Tamil, Tanglish & English)
    years_back = 10
    time_range = None
    target_year = None

    m_hist_n = re.search(
        r"(?:கடந்த|முந்தைய|last|past)\s+(\d+)\s*(?:ஆண்டுகளில்|ஆண்டுகளின்|ஆண்டுகள்|வருடங்களில்|வருடங்கள்|years?)",
        q,
        re.IGNORECASE,
    )
    m_last_single_yr = re.search(
        r"(?:கடந்த\s+ஆண்டு|கடந்த\s+வருடம்|போன\s+ஆண்டு|போன\s+வருடம்|போன\s+வருஷம்|last\s+year|previous\s+year)",
        q,
        re.IGNORECASE,
    )
    m_this_yr = re.search(
        r"(?:இந்த\s+ஆண்டு|இந்த\s+வருடம்|நடப்பு\s+ஆண்டு|this\s+year|current\s+year)",
        q,
        re.IGNORECASE,
    )
    m_few_yrs = re.search(
        r"(?:கடந்த\s+சில\s+ஆண்டுகளில்|கடந்த\s+சில\s+வருடங்களில்|past\s+few\s+years|last\s+few\s+years)",
        q,
        re.IGNORECASE,
    )
    has_yearly_grouping = bool(
        re.search(
            r"(?:ஒவ்வொரு\s+ஆண்டும்|ஆண்டுதோறும்|ஆண்டு\s*வாரியாக|வருட\s*வாரியாக|வருடந்தோறும்|ஒவ்வொரு\s+வருடமும்|each\s+year|every\s+year|year\s+by\s+year|yearly|annual)",
            q,
            re.IGNORECASE,
        )
    )

    if m_hist_n:
        years_back = max(1, min(50, int(m_hist_n.group(1))))
        time_range = f"LAST_{years_back}_YEARS"
    elif m_last_single_yr:
        years_back = 1
        target_year = cur_year - 1
        time_range = "LAST_YEAR"
    elif m_this_yr:
        years_back = 1
        target_year = cur_year
        time_range = "THIS_YEAR"
    elif m_few_yrs:
        years_back = 5
        time_range = "LAST_5_YEARS"
    elif has_yearly_grouping:
        years_back = 10
        time_range = "YEAR_BY_YEAR"

    forecast_horizon = 0
    has_forecast = bool(
        re.search(
            r"(?:அடுத்த|வரும்|எதிர்கால|கணிப்பு|கணிப்பை|முன்கணிப்பு|எப்படி\s+இருக்கும்|eppadi\s+irukkum|forecast|predict|projection|next\s+\d+\s+years)",
            q,
            re.IGNORECASE,
        )
    )
    if has_forecast:
        m_f = re.search(
            r"(?:அடுத்த|வரும்|next|upcoming)\s+(\d+)\s*(?:ஆண்டுகளுக்கு|ஆண்டுகளில்|ஆண்டுகள்|வருடங்கள்|years?)",
            q,
            re.IGNORECASE,
        )
        if m_f:
            forecast_horizon = int(m_f.group(1))
            time_range = f"NEXT_{forecast_horizon}_YEARS"
        elif re.search(r"(?:அடுத்த\s+ஆண்டு|அடுத்த\s+வருடம்|next\s+year)", q, re.IGNORECASE):
            forecast_horizon = 1
            time_range = "NEXT_YEAR"
        else:
            forecast_horizon = years_back or 10
            time_range = f"NEXT_{forecast_horizon}_YEARS"

    has_time_expr = bool(time_range or has_yearly_grouping or has_forecast)
    grouping = "YEAR" if (has_yearly_grouping or m_hist_n or m_few_yrs or has_forecast) else None

    # 3. Extract Family Card & Person Entity BEFORE deciding intent (Sections 3, 4, 5, 6, 7, 15)
    m_card = re.search(r"\b([A-Z]{2,5}/\d{1,5})\b", q, re.IGNORECASE)
    family_card = m_card.group(1).upper() if m_card else None

    person_entity = extract_person_entity_from_multilingual_query(q)
    person_name = person_entity.get("transliterated_name") or person_entity.get("original_name")

    # 4. Semantic Signals
    has_trend = bool(
        re.search(
            r"(?:மாறியுள்ளது|மாற்றம்|மாற்றத்தை|போக்கு|வளர்ச்சி|குறைவு|அதிகரிப்பு|eppadi\s+change|change\s+aachu|trend|changed|growth)",
            q,
            re.IGNORECASE,
        )
    )
    has_stats = bool(
        re.search(
            r"(?:புள்ளிவிவரம்|புள்ளிவிவரங்கள்|சராசரி|statistics|statistical|summary)",
            q,
            re.IGNORECASE,
        )
    )
    has_chart_request = bool(
        re.search(
            r"(?:வரைபடம்|வரைபடத்துடன்|வரைபடமாக|வரைபடத்தில்|வரைபட|சார்ட்|கிராப்|chart|charts|graph|plot|visualize|visualise)",
            q,
            re.IGNORECASE,
        )
    )
    has_comparison = bool(re.search(r"(?:ஒப்பிடு|ஒப்பீடு|compare|comparison|vs\b|versus)", q, re.IGNORECASE)) or len(metrics) >= 2
    has_status = bool(re.search(r"(?:நிலை\s+என்ன|நிலை|status|விவரம்|விவரங்கள்|details|enna)", q, re.IGNORECASE))
    has_family = bool(
        re.search(
            r"(?:குடும்பம்|குடும்பத்தில்|குடும்பத்தின்|குடும்ப|உறுப்பினர்கள்|உறுப்பினர்களை|family|kudumbam|kudumbathil)",
            q,
            re.IGNORECASE,
        )
    )
    has_count = bool(
        re.search(
            r"(?:எண்ணிக்கை|எண்ணிக்கையைத்|எண்ணிக்கையை|எத்தனை|மொத்தம்|how\s+many|count|evlo|evalo|ethanai|eththanai)",
            q,
            re.IGNORECASE,
        )
    )
    has_who_members = bool(
        re.search(
            r"(?:யார்\s+யார்|யார்|உறுப்பினர்கள்\s+யார்|உறுப்பினர்களை\s+காட்டு|yaar\s+yaar|yaar|who\s+are|members\s+list)",
            q,
            re.IGNORECASE,
        )
    )
    has_contact = bool(
        re.search(
            r"(?:தொலைபேசி|அலைபேசி|தொடர்பு\s+எண்|போன்|முகவரி|phone|mobile|contact|address)",
            q,
            re.IGNORECASE,
        )
    )
    has_list = bool(re.search(r"(?:காட்டு|பட்டியலிடுக|பட்டியல்|தரவும்|கொடு|விளக்கவும்|list|show|kudu|kaatu)", q, re.IGNORECASE))

    # 5. PRIORITY INTENT RESOLUTION (Sections 4, 5, 6, 7, 8, 9, 10)
    # CRITICAL RULE: If a specific Person or Family Card is present, NEVER classify as SACRAMENT_STATISTICS!
    semantic_intent = "GENERAL_DATABASE_QUERY"
    langgraph_intent = "GENERAL_DATABASE_QUERY"
    scope = "GENERAL_MEMBER"
    requested_info = "general_information"
    db_operation = "SELECT_RECORDS"

    if person_name or family_card:
        entity_type = "FAMILY_CARD" if family_card else ("FAMILY" if has_family else "PERSON")
        entity_val = family_card or person_name
        has_sacrament_any = bool(metrics or "திருவருட்சாதன" in q or "அருட்சாதன" in q or "sacrament" in q_low)
        is_all_sacs_req = bool(
            re.search(r"\b(?:all\s+sacraments?|all\s+sacramental)\b", q_low)
            or any(k in q for k in ["அனைத்து திருவருட்சாதன", "அனைத்து அருட்சாதன", "எல்லா திருவருட்சாதன"])
            or (has_sacrament_any and not metrics)
        )

        if has_family and has_sacrament_any:
            # FAMILY + SACRAMENT COMBINATION QUERY RULE:
            # Sacrament determines WHAT information is requested; Family determines WHO (target_scope = FAMILY_MEMBERS)
            target_scope = "FAMILY_MEMBERS"
            langgraph_intent = "SACRAMENT_SEARCH"
            if is_all_sacs_req or len(metrics) > 1:
                semantic_intent = "FAMILY_ALL_SACRAMENTS"
                scope = "FAMILY_ALL_SACRAMENTS"
                requested_info = "ALL_SACRAMENTS"
                db_operation = "LOOKUP_FAMILY_ALL_SACRAMENTS"
                primary_sacrament_code = "ALL_SACRAMENTS"
            else:
                semantic_intent = "FAMILY_SACRAMENT_RECORDS"
                requested_info = "SACRAMENT_RECORDS"
                db_operation = "LOOKUP_FAMILY_SACRAMENT_RECORDS"
                if primary_sacrament_code == "BAPTISM":
                    scope = "FAMILY_BAPTISM_RECORDS"
                elif primary_sacrament_code == "FIRST_HOLY_COMMUNION":
                    scope = "FAMILY_COMMUNION_RECORDS"
                elif primary_sacrament_code == "CONFIRMATION":
                    scope = "FAMILY_CONFIRMATION_RECORDS"
                elif primary_sacrament_code == "MARRIAGE":
                    scope = "FAMILY_MARRIAGE_RECORDS"
                elif primary_sacrament_code == "DEATH":
                    scope = "FAMILY_DEATH_RECORDS"
                else:
                    scope = "FAMILY_ALL_SACRAMENTS"
        elif has_family and has_count and not metrics:
            # Section 5: FAMILY_MEMBER_COUNT ("Antony Selvan குடும்பம் எத்தனை நபர்கள் உள்ளார்கள்")
            semantic_intent = "FAMILY_MEMBER_COUNT"
            langgraph_intent = "FAMILY_SEARCH"
            scope = "FAMILY_MEMBER_COUNT"
            target_scope = "FAMILY_MEMBERS"
            requested_info = "MEMBER_COUNT"
            db_operation = "COUNT_FAMILY_MEMBERS"
            primary_sacrament_code = None
            metrics = []
        elif has_family and (has_who_members or ("உறுப்பினர்" in q and not has_count) or "members" in q_low):
            # Section 6: FAMILY_MEMBER_LIST ("Antony Selvan குடும்பத்தில் யார் யார் இருக்கிறார்கள்?")
            semantic_intent = "FAMILY_MEMBER_LIST"
            langgraph_intent = "FAMILY_SEARCH"
            scope = "FAMILY_MEMBERS_ONLY"
            target_scope = "FAMILY_MEMBERS"
            requested_info = "MEMBER_LIST"
            db_operation = "LIST_FAMILY_MEMBERS"
            primary_sacrament_code = None
            metrics = []
        elif has_contact:
            semantic_intent = "MEMBER_CONTACT"
            langgraph_intent = "MEMBER_SEARCH"
            scope = "MEMBER_ADDRESS" if ("முகவரி" in q or "address" in q_low) else "MEMBER_PHONE"
            target_scope = "SINGLE_PERSON"
            requested_info = "MEMBER_CONTACT"
            db_operation = "LOOKUP_MEMBER_CONTACT"
            primary_sacrament_code = None
            metrics = []
        elif has_sacrament_any:
            # Section 7: PERSON_SACRAMENT_STATUS ("Antony Selvan அவர்களின் ஞானஸ்நான நிலை என்ன?", "Show Cathrine Suganya's baptism status.")
            semantic_intent = "PERSON_SACRAMENT_STATUS"
            langgraph_intent = "SACRAMENT_SEARCH"
            target_scope = "SINGLE_PERSON"
            if is_all_sacs_req:
                scope = "ALL_SACRAMENTS"
            elif primary_sacrament_code == "BAPTISM":
                scope = "BAPTISM_STATUS"
            elif primary_sacrament_code == "FIRST_HOLY_COMMUNION":
                scope = "COMMUNION_STATUS"
            elif primary_sacrament_code == "CONFIRMATION":
                scope = "CONFIRMATION_STATUS"
            elif primary_sacrament_code == "MARRIAGE":
                scope = "MARRIAGE_STATUS"
            elif primary_sacrament_code == "DEATH":
                scope = "DEATH_STATUS"
            else:
                scope = "ALL_SACRAMENTS"
            requested_info = "SACRAMENT_STATUS"
            db_operation = "LOOKUP_MEMBER_SACRAMENT"
        elif family_card or has_family:
            semantic_intent = "FAMILY_CARD_LOOKUP" if family_card else "FAMILY_DETAILS"
            langgraph_intent = "FAMILY_SEARCH"
            scope = "FAMILY_DETAILS"
            target_scope = "FAMILY_MEMBERS"
            requested_info = "FAMILY_DETAILS"
            db_operation = "RETRIEVE_FAMILY_BY_ID"
        else:
            semantic_intent = "PERSON_DETAILS"
            langgraph_intent = "MEMBER_SEARCH"
            scope = "FAMILY_DETAILS"
            target_scope = "SINGLE_PERSON"
            requested_info = "PERSON_DETAILS"
            db_operation = "LOOKUP_PERSON"
    else:
        entity_type = "SACRAMENT" if primary_sacrament_code else "PARISH"
        entity_val = primary_sacrament_code or "AUTHORIZED_PARISH"
        target_scope = "PARISH"

        if has_comparison and (has_time_expr or has_trend or has_stats):
            semantic_intent = "COMPARISON"
            langgraph_intent = "COMPARISON"
            requested_info = "multi_sacrament_comparison"
            db_operation = "AGGREGATE_MULTI_SACRAMENT_SERIES"
        elif has_forecast:
            semantic_intent = "FORECAST"
            langgraph_intent = "FORECAST"
            requested_info = "sacrament_forecast"
            db_operation = "FORECAST_YEARLY_SERIES"
        elif has_trend and (has_time_expr or metrics):
            semantic_intent = "TREND_ANALYSIS"
            langgraph_intent = "TREND_ANALYSIS"
            requested_info = "trend_analysis"
            db_operation = "ANALYZE_YEARLY_TREND"
        elif (m_last_single_yr or m_this_yr) and metrics and not has_trend and not has_forecast:
            # Single year sacrament count ("கடந்த ஆண்டு மொத்தம் எத்தனை ஞானஸ்நானம்ச் சடங்குகள் நமது பங்கில் நடைபெற்றது")
            semantic_intent = "SACRAMENT_STATISTICS"
            langgraph_intent = "COUNT"
            requested_info = "single_year_sacrament_count"
            db_operation = "COUNT_SACRAMENT_BY_YEAR"
        elif (has_time_expr or has_stats or has_chart_request) and metrics:
            # Section 8: Multi-year SACRAMENT_STATISTICS ("கடந்த 10 ஆண்டுகளில் எத்தனை ஞானஸ்நானங்கள் நடந்துள்ளன?")
            semantic_intent = "SACRAMENT_STATISTICS"
            langgraph_intent = "HISTORICAL_ANALYSIS"
            requested_info = "yearly_sacrament_statistics"
            db_operation = "AGGREGATE_SACRAMENT_BY_YEAR"
        elif has_count and not metrics:
            if has_family:
                semantic_intent = "FAMILY_COUNT"
                langgraph_intent = "COUNT"
                scope = "COUNT_FAMILIES"
                requested_info = "parish_family_count"
                db_operation = "COUNT_PARISH_FAMILIES"
            else:
                semantic_intent = "MEMBER_COUNT"
                langgraph_intent = "COUNT"
                scope = "COUNT_MEMBERS"
                requested_info = "parish_member_count"
                db_operation = "COUNT_PARISH_MEMBERS"
        elif has_count and metrics:
            semantic_intent = "SACRAMENT_STATISTICS"
            langgraph_intent = "COUNT"
            requested_info = "sacrament_count"
            db_operation = "COUNT_SACRAMENT_RECORDS"
        elif has_list:
            semantic_intent = "PARISH_STATISTICS"
            langgraph_intent = "LIST"
            scope = "LIST_FAMILIES" if has_family else "LIST_MEMBERS"
            requested_info = "list_records"
            db_operation = "LIST_PARISH_RECORDS"

    query_decomposition = {
        "intent": semantic_intent,
        "person": person_name,
        "family_reference": (person_name or family_card) if (has_family or family_card) else None,
        "target_scope": target_scope,
        "requested_information": requested_info,
        "sacrament": primary_sacrament_code,
        "time_range": time_range,
    }

    structured_repr = (
        f"INTENT={semantic_intent} | LG_INTENT={langgraph_intent} | LANGUAGE={lang} "
        f"| ENTITY_TYPE={entity_type} | ENTITY={entity_val or 'NONE'} "
        f"| TARGET_SCOPE={target_scope} | SACRAMENT={primary_sacrament_code or 'NONE'} "
        f"| TIME_RANGE={time_range or 'NONE'} | SCOPE={scope} | DB_OP={db_operation}"
    )

    return {
        "original_query": q,
        "detected_language": "ta" if lang in ("ta", "tanglish") and is_tamil(q) else lang,
        "language": lang,
        "normalized_query": structured_repr,
        "intent_query": semantic_intent,
        "entity_query": entity_val or "NONE",
        "intent": langgraph_intent,
        "semantic_intent": semantic_intent,
        "entity_type": entity_type,
        "entity": entity_val,
        "target_scope": target_scope,
        "requested_information": requested_info,
        "database_operation": db_operation,
        "query_decomposition": query_decomposition,
        "sacrament": primary_sacrament_code,
        "metrics": metrics,
        "canonical_terms": canonical_terms,
        "time_range": time_range,
        "period": time_range,
        "target_year": target_year,
        "years_back": years_back,
        "forecast_horizon": forecast_horizon,
        "grouping": grouping,
        "chart_requested": has_chart_request,
        "family_card": family_card,
        "person_name": person_name,
        "person_entity": person_entity,
        "scope": scope,
        "sub_intent": scope if langgraph_intent in ("COUNT", "LIST") else None,
        "final_response_language": "ta" if is_tamil(q) else "en",
    }


def normalize_tamil_query(query_text: str) -> str:
    """
    Preserves the original Tamil query untouched (Sections 1 & 2:
    Never overwrite the original query with a translated or Tanglish sentence).
    """
    return query_text
