"""
Sacrament Normalizer & Speech Recognition Error Correction Layer
================================================================
Normalizes misspellings, phonetic deviations, speech-to-text transcription errors,
and Tamil/Tanglish dialectal variations for all Catholic sacraments before RAG retrieval.
"""

import re
import difflib

# ── Canonical Sacrament Definitions & Aliases ──────────────────────────────────
SACRAMENT_MAPPINGS = {
    "Holy Communion": {
        "canonical_en": "First Holy Communion",
        "canonical_ta": "முதல் நற்கருணை",
        "canonical_ta_short": "புதுநன்மை",
        "canonical_tanglish": "First Holy Communion",
        "field_name": "fhc_date",
        "table_name": "tabMember",
        "aliases": [
            # Tanglish / English Speech Variations & Errors
            "pothunamai", "puthunamai", "pothunanmai", "puthunanmai", "pothunmai", 
            "puthunmai", "puthunaamai", "pothunaamai", "puthunanmey", "pothunanmey", 
            "puthunamay", "pothunamay", "puthunama", "pothunama", "puthumai", "pothumai",
            "pothunanmay", "putunanmai", "potunanmai", "pudunanmai", "podunanmai",
            "pothu nanmai", "puthu nanmai", "pothu namai", "puthu namai",
            "first holy communion", "holy communion", "first communion", "communion", 
            "eucharist", "fhc", "eucharistia", "cummunion", "comunion", "communiyan",
            "first holy comunion", "first holy cummunion", "f.h.c", "f h c",
            "holy comunnion", "first comunnion",
            
            # Tamil Speech-to-Text Misrecognitions (including "பொதுநன்மை" from STT)
            "பொதுநன்மை", "பொது நன்மை", "பொதுநன்மை நிகழ்வுகள்", "பொது நன்மை நிகழ்வுகள்",
            "பொதுநன்மை நிகழ்வு", "பொது நன்மை நிகழ்வு",
            "போதுநன்மை", "போது நன்மை", "போதுநன்மை நிகழ்வுகள்",
            "புதுநன்மை", "புது நன்மை", "புதுநன்மை நிகழ்வுகள்", "புது நன்மை நிகழ்வுகள்",
            "புதுநன்மை நிகழ்வு", "புது நன்மை நிகழ்வு",
            "நற்கருணை", "நற்கருனை", "நற் கருணை",
            "முதல் நற்கருணை", "முதல் நற்கருனை", "முதல்நற்கருணை", "முதல்நற்கருனை",
            "நற்கருணைப்பரிசு", "திவ்விய நற்கருணை", "திவ்ய நற்கருணை", "நற்கருணை விருந்து",
            "கம்யூனியன்", "கம்யூன்யன்"
        ]
    },
    "Baptism": {
        "canonical_en": "Baptism",
        "canonical_ta": "திருமுழுக்கு",
        "canonical_ta_short": "ஞானஸ்நானம்",
        "canonical_tanglish": "Thirumuzhukku",
        "field_name": "bapt_date",
        "table_name": "tabMember",
        "aliases": [
            "baptism", "baptisam", "baptisum", "babtism", "babtisum", "babtizam",
            "bapthisam", "bapthisum", "bapt", "babt", "bap", "christening", "kristening",
            "gnanasnanam", "nanasnanam", "gnanasnanamu", "gnana snanam", "nana snanam",
            "gnanasnaanam", "nanasnaanam",
            "thirumulukku", "thirumuzhukku", "thirumuluku", "thirumuzhuku", "thirumulku",
            "ஞானஸ்நானம்", "ஞானஸ்னானம்", "ஞானஸ்நான", "ஞான ஸ்நானம்", "ஞான ஸ்னானம்",
            "ஞானஸ்தானம்", "ஞான ஸ்தானம்", "ஞானஸ்நானப் பதிவு",
            "திருமுழுக்கு", "திருமுழுகு", "திரு முழுக்கு", "திருமுழுக்குகள்", "திருமுழுக்கின்",
            "திருமுழுக்கு பெற்ற", "திருமுழுக்குப் பதிவு", "திருமுழுக்கு நிகழ்வு"
        ]
    },
    "Confirmation": {
        "canonical_en": "Confirmation",
        "canonical_ta": "உறுதிப்பூசுதல்",
        "canonical_ta_short": "உறுதிப்பூசுதல்",
        "canonical_tanglish": "Uruthipoosuthal",
        "field_name": "cnf_date",
        "table_name": "tabMember",
        "aliases": [
            "confirmation", "confrmation", "comfirmation", "confirmasan", "confirmashun",
            "confirm", "conf", "cnf", "chrism", "chrisam",
            "uruthipoosuthal", "uruthipoothal", "uruthipoosudhal", "uruthi poosuthal",
            "uruthipoosuthall", "urudhipoosuthal",
            "thidapaduthal", "thidappaduthal", "thida paduthal",
            "உறுதிப்பூசுதல்", "உறுதிபூசுதல்", "உறுதி பூசுதல்", "உறுதிப் பூசுதல்",
            "உறுதிப்படுத்தல்", "உறுதி படுத்தல்", "திடப்படுத்தல்", "திட படுத்தல்"
        ]
    },
    "Marriage": {
        "canonical_en": "Marriage",
        "canonical_ta": "திருமணம்",
        "canonical_ta_short": "திருமணம்",
        "canonical_tanglish": "Thirumanam",
        "field_name": "mrg_date",
        "table_name": "tabMember",
        "aliases": [
            "marriage", "marrage", "marrige", "marryage", "marr", "wedding", "matrimony",
            "thirumanam", "tirumanam", "thiru manam", "kalyanam", "kalyaanam", "kalyana",
            "vivaham", "vivaaham", "vivaha",
            "திருமணம்", "திருமண", "திரு மணம்", "கல்யாணம்", "கல்யாண",
            "விவாகம்", "விவாக", "மணமக்கள்", "திருமணப் பதிவு", "திருமண நிகழ்வு"
        ]
    },
    "Death": {
        "canonical_en": "Death",
        "canonical_ta": "இறப்பு",
        "canonical_ta_short": "இறப்பு",
        "canonical_tanglish": "Irappu",
        "field_name": "death_date",
        "table_name": "tabMember",
        "aliases": [
            "death", "died", "deceased", "burial", "funeral", "cemetery", "graveyard",
            "irappu", "irappu pathivu", "maranam", "adakkam", "adakkachadangu",
            "kallara", "kallarai", "eemachadangu", "kaithoosi",
            "இறப்பு", "இறந்த", "மரணம்", "மரண", "அடக்கம்", "அடக்கச்சடங்கு",
            "கல்லறை", "கல்லறைப் பதிவு", "ஈமச்சடங்கு", "இறுதிச்சடங்கு"
        ]
    },
    "Anointing of the Sick": {
        "canonical_en": "Anointing of the Sick",
        "canonical_ta": "நோயாளரின் பூசுதல்",
        "canonical_ta_short": "நோயில்பூசுதல்",
        "canonical_tanglish": "Noyilpoosuthal",
        "field_name": "anointing_date",
        "table_name": "tabAnointing Of Sick",
        "aliases": [
            "anointing", "anointing of the sick", "extreme unction", "last rites",
            "noyil poosuthal", "noyilpoosuthal", "noyalargalin poosuthal",
            "noyalargal poosuthal", "kadaisi poosuthal",
            "நோயில்பூசுதல்", "நோயில் பூசுதல்", "நோயாளரின் பூசுதல்", "நோயாளிகள் பூசுதல்",
            "கடைசி பூசுதல்", "நோயில் பூசுதல் சடங்கு"
        ]
    }
}

def is_tamil_script(text: str) -> bool:
    """Checks whether the text contains Tamil unicode characters."""
    return any('\u0B80' <= c <= '\u0BFF' for c in text)

def _phonetic_key(s: str) -> str:
    """Computes a simplified phonetic key for Indian English/Tanglish words."""
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9]', '', s)
    s = s.replace('th', 't').replace('ph', 'f').replace('dh', 'd').replace('zh', 'l')
    s = s.replace('oo', 'u').replace('ee', 'i').replace('aa', 'a').replace('ou', 'u')
    s = s.replace('o', 'u')  # Normalizes pothu -> puthu
    # Collapse duplicate characters
    s = re.sub(r'(.)\1+', r'\1', s)
    return s

def normalize_sacrament_query(query_text: str) -> tuple[str, dict]:
    """
    Normalizes sacrament misspellings, STT errors, and phonetic variations.
    
    Returns:
        (normalized_query, info_dict)
    """
    if not query_text or not isinstance(query_text, str):
        return query_text, {}

    original_text = query_text.strip()
    normalized_text = original_text
    detected_sacrament = None
    detected_raw = None
    corrected_term = None

    is_tamil = is_tamil_script(original_text)

    # 1. Exact Phrase & Multi-word Alias Matching (Longest aliases first)
    all_alias_entries = []
    for sac_key, sac_data in SACRAMENT_MAPPINGS.items():
        for alias in sac_data["aliases"]:
            all_alias_entries.append((alias, sac_key, sac_data))
    
    # Sort aliases descending by length so longer phrases match first
    all_alias_entries.sort(key=lambda x: len(x[0]), reverse=True)

    for alias, sac_key, sac_data in all_alias_entries:
        # Build regex with boundary checking
        if is_tamil_script(alias):
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
        else:
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)

        m = pattern.search(normalized_text)
        if m:
            detected_raw = m.group(0)
            detected_sacrament = sac_key
            # Choose canonical term matching query's language
            if is_tamil:
                # If user used a form of puthunanmai (like பொதுநன்மை), normalize to புதுநன்மை / நற்கருணை
                if any(k in detected_raw for k in ["நன்மை", "நமை", "னமை", "புது", "பொது", "போது"]) or any(k in alias.lower() for k in ["puthu", "pothu", "nanmai", "namai"]):
                    corrected_term = sac_data["canonical_ta_short"] # e.g. புதுநன்மை
                else:
                    corrected_term = sac_data["canonical_ta"]
            else:
                if any(k in detected_raw.lower() for k in ["puthu", "pothu", "nanmai", "namai", "mai"]):
                    corrected_term = sac_data["canonical_tanglish"] # e.g. Puthunanmai
                else:
                    corrected_term = sac_data["canonical_en"]

            normalized_text = pattern.sub(corrected_term, normalized_text, count=1)
            break

    # 2. Token-level Fuzzy & Phonetic Matching (if not detected via direct alias)
    if not detected_sacrament:
        # Split into words while keeping punctuation
        words = re.findall(r'[\w\u0B80-\u0BFF]+|[^\w\s\u0B80-\u0BFF]+|\s+', original_text)
        
        for idx, token in enumerate(words):
            token_clean = token.strip()
            if len(token_clean) < 4:
                continue

            token_phone = _phonetic_key(token_clean) if not is_tamil_script(token_clean) else ""

            best_match = None
            best_score = 0.0

            for sac_key, sac_data in SACRAMENT_MAPPINGS.items():
                for alias in sac_data["aliases"]:
                    # Check Tanglish phonetic key
                    if token_phone and not is_tamil_script(alias):
                        alias_phone = _phonetic_key(alias)
                        if token_phone == alias_phone or (len(token_phone) >= 4 and token_phone in alias_phone):
                            best_match = (sac_key, sac_data, alias)
                            best_score = 0.95
                            break

                    # Check Levenshtein ratio
                    ratio = difflib.SequenceMatcher(None, token_clean.lower(), alias.lower()).ratio()
                    if ratio > best_score and ratio >= 0.80:
                        best_score = ratio
                        best_match = (sac_key, sac_data, alias)

                if best_score >= 0.90:
                    break

            if best_match and best_score >= 0.80:
                sac_key, sac_data, matched_alias = best_match
                detected_raw = token_clean
                detected_sacrament = sac_key

                if is_tamil:
                    corrected_term = sac_data["canonical_ta_short"] if "நன்மை" in matched_alias else sac_data["canonical_ta"]
                else:
                    corrected_term = sac_data["canonical_tanglish"] if "puthu" in matched_alias.lower() or "pothu" in matched_alias.lower() else sac_data["canonical_en"]

                words[idx] = corrected_term
                normalized_text = "".join(words)
                break

    # Required Logging
    if detected_sacrament and detected_raw != corrected_term:
        print(f"[Sacrament Normalizer] Original STT text: '{original_text}'")
        print(f"[Sacrament Normalizer] Detected Sacrament: '{detected_raw}' -> Canonical: '{detected_sacrament}' ({corrected_term})")
        print(f"[Sacrament Normalizer] Corrected Term: '{corrected_term}'")
        print(f"[Sacrament Normalizer] Final RAG query: '{normalized_text}'")

    info = {
        "original_text": original_text,
        "normalized_text": normalized_text,
        "detected_sacrament": detected_sacrament,
        "detected_raw": detected_raw,
        "corrected_term": corrected_term,
        "was_normalized": bool(detected_sacrament and original_text != normalized_text)
    }

    return normalized_text, info
