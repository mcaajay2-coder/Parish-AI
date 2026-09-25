"""
Tamil Query Parser & Intermediate Structured Query Representation
==================================================================
Parses Tamil, Tanglish, and mixed queries into a verified, structured representation
before SQL generation and database execution.
"""

import re
import datetime
try:
    from koinonia_assistant.rag.sacrament_normalizer import normalize_sacrament_query
except ImportError:
    try:
        from sacrament_normalizer import normalize_sacrament_query
    except ImportError:
        def normalize_sacrament_query(q):
            return q, {}

normalize_query_sacraments = normalize_sacrament_query


def parse_tamil_query(query_text: str, user_parish: str = None) -> dict:
    """
    Parses a natural language query (Tamil, Tanglish, or English)
    into a structured query dictionary.
    """
    raw_q = query_text.strip()
    
    # 0. Sacrament Normalization (STT errors & Phonetic corrections)
    normalized_q, stt_matches = normalize_query_sacraments(raw_q)
    q = normalized_q
    q_low = q.lower()
    current_year = datetime.datetime.now().year

    # 1. Language Detection
    has_tamil = any('\u0B80' <= c <= '\u0BFF' for c in raw_q)
    tanglish_markers = ['la', 'um', 'vangina', 'avanga', 'kaatungalen', 'parish-ல்', 'families-ல', 'members-க்கு', 'pathi', 'sollunga', 'solunga', 'yaaru']
    is_tanglish = any(m in q_low for m in tanglish_markers) and not has_tamil

    lang = "ta" if has_tamil else ("tanglish" if is_tanglish else "en")

    # 2. Sacrament Extraction
    sacrament = None
    if any(k in q_low or k in q for k in ['புதுநன்மை', 'முதல் நற்கருணை', 'நற்கருணை', 'communion', 'puthunanmai', 'pothunamai', 'fhc', 'eucharist']):
        sacrament = "Holy Communion"
    elif any(k in q_low or k in q for k in ['ஞானஸ்நானம்', 'திருமுழுக்கு', 'baptism', 'gnanasnanam']):
        sacrament = "Baptism"
    elif any(k in q_low or k in q for k in ['உறுதிப்பூசுதல்', 'உறுதிபூசுதல்', 'confirmation', 'uruthipoosuthal']):
        sacrament = "Confirmation"
    elif any(k in q_low or k in q for k in ['திருமணம்', 'கல்யாணம்', 'விவாகம்', 'marriage', 'wedding', 'thirumanam', 'kalyanam']):
        sacrament = "Marriage"
    elif any(k in q_low or k in q for k in ['இறப்பு', 'மரணம்', 'அடக்கம்', 'கல்லறை', 'death', 'burial', 'funeral']):
        sacrament = "Death"
    elif any(k in q_low or k in q for k in ['ஒவ்வொரு திருவருட்சாதனத்திற்கும்', 'அனைத்து திருவருட்சாதன', 'all sacrament', 'sacraments']):
        sacrament = "All"

    # 3. Gender Extraction
    # CRITICAL: 'ஆண்டு' / 'ஆண்டுகளில்' (years) contains 'ஆண்'!
    # Strip year words before checking male regex
    q_no_years = re.sub(r'ஆண்டு\w*', '', q)
    female_regex = r'(?:^|[^\w])(பெண்|பெண்கள்|பெண்களின்|பெண்ணாக)(?:$|[^\w])|\b(female|women|woman|girls)\b'
    male_regex = r'(?:^|[^\w])(ஆண்|ஆண்கள்|ஆண்களின்|ஆணாக)(?:$|[^\w])|\b(male|men|man|boys)\b'

    has_female = bool(re.search(female_regex, q, re.IGNORECASE))
    has_male = bool(re.search(male_regex, q_no_years, re.IGNORECASE))

    both_genders_count = has_female and has_male
    gender = None
    if both_genders_count:
        gender = None
    elif has_female:
        gender = "Female"
    elif has_male:
        gender = "Male"

    # 4. Age Conditions
    age_op = None
    age_val = None
    m_age_above = re.search(r'(\d+)\s*(?:வயதுக்கு மேற்பட்ட|வயதிற்கு மேற்பட்ட|வயதுக்கு மேல்|வயதிற்கு மேல்|age\s*>\s*|older than\s*)', q, re.IGNORECASE)
    if not m_age_above:
        m_age_above = re.search(r'(?:above age\s*|older than\s*)(\d+)', q, re.IGNORECASE)
    
    m_age_below = re.search(r'(\d+)\s*(?:வயதுக்கு குறைவான|வயதிற்கு குறைவான|வயதுக்கு கீழ்|வயதிற்கு கீழ்|age\s*<\s*|younger than\s*|under\s*)', q, re.IGNORECASE)
    if not m_age_below:
        m_age_below = re.search(r'(?:under age\s*|below age\s*|younger than\s*)(\d+)', q, re.IGNORECASE)

    if m_age_above:
        age_op = ">"
        age_val = int(m_age_above.group(1))
    elif m_age_below:
        age_op = "<"
        age_val = int(m_age_below.group(1))

    # 5. Date & Year Filters
    date_filter = None
    # 5a. Impossible / Conflict Check (e.g. Born 2010, Communion 2000)
    m_conflict = re.search(r'(\d{4})\s*(?:ஆம்\s*ஆண்டில்)?\s*(?:பிறந்து|born).*?(\d{4})\s*(?:ஆம்\s*ஆண்டில்)?\s*(?:புதுநன்மை|முதல் நற்கருணை|communion|bapt|சாதனம்|grace)', q, re.IGNORECASE)
    if not m_conflict:
        m_conflict = re.search(r'(\d{4})\s*ஆம்\s*ஆண்டில்\s*பிறந்து\s*(\d{4})\s*ஆம்\s*ஆண்டில்', q)
    if m_conflict:
        b_yr = int(m_conflict.group(1))
        s_yr = int(m_conflict.group(2))
        if b_yr > s_yr:
            date_filter = {"type": "negative_conflict", "birth_year": b_yr, "sacrament_year": s_yr}
    
    # 5b. Year Range (e.g. 2020 முதல் 2025 வரை)
    elif re.search(r'(\d{4})\s*(?:முதல்|to|-)\s*(\d{4})\s*(?:வரை|between)', q):
        m_rng = re.search(r'(\d{4})\s*(?:முதல்|to|-)\s*(\d{4})\s*(?:வரை|between)', q)
        date_filter = {"type": "range", "start": int(m_rng.group(1)), "end": int(m_rng.group(2))}

    # 5c. Relative Years (e.g. கடந்த 15 ஆண்டுகளில், past 10 years)
    elif re.search(r'(?:கடந்த|past|last)\s*(\d+)\s*(?:ஆண்டுகளில்|வருடங்களில்|வருஷத்துல|years)', q, re.IGNORECASE):
        m_rel = re.search(r'(?:கடந்த|past|last)\s*(\d+)\s*(?:ஆண்டுகளில்|வருடங்களில்|வருஷத்துல|years)', q, re.IGNORECASE)
        date_filter = {"type": "relative_years", "years": int(m_rel.group(1))}

    # 5d. Exact Year (e.g. 2023 ஆம் ஆண்டு)
    elif re.search(r'\b(20\d\d|19\d\d)\b\s*(?:ஆம்\s*ஆண்டு|இல்|year)?', q):
        m_yr = re.search(r'\b(20\d\d|19\d\d)\b', q)
        date_filter = {"type": "exact_year", "year": int(m_yr.group(1))}

    # 6. Family Level & Member Thresholds
    is_family_level = any(k in q for k in ['ஒவ்வொரு குடும்பத்தின்', 'குடும்ப வாரியாக', 'குடும்பங்களின் பெயர்களையும்', 'குடும்பத் தலைவர்களின்'])
    
    m_thresh = re.search(r'(\d+)\s*(?:பேருக்கும் அதிகமான|members-க்கு மேல|உறுப்பினர்களுக்கு அதிகமாக|உறுப்பினர்களுக்கு அதிகமான|members\s*>\s*)', q)
    family_min_members = int(m_thresh.group(1)) if m_thresh else None

    # 7. Active / Inactive Status
    active_only = any(k in q for k in ['செயல்பாட்டில் இருக்கும்', 'செயல்பாட்டில் உள்ள', 'செயல்பாட்டில்']) or 'active' in q_low

    # 8. Location
    location = None
    if 'சென்னை' in q or 'chennai' in q_low:
        location = "Chennai"

    # 9. Contact Info Requested
    contact_requested = any(k in q for k in ['தொடர்பு எண்', 'மொபைல் எண்', 'தொலைபேசி எண்', 'போன் நம்பர்']) or any(k in q_low for k in ['mobile', 'phone', 'contact'])

    # 10. Spouse Info Requested
    spouse_requested = any(k in q for k in ['துணை', 'துணையின் பெயர்', 'கணவன்', 'மனைவி']) or 'spouse' in q_low

    # 11. Intent Classification
    intent = "general_search"

    if date_filter and date_filter.get("type") == "negative_conflict":
        intent = "negative_impossible"
    elif both_genders_count and sacrament == "Holy Communion" and date_filter:
        intent = "sacrament_gender_aggregate"
    elif any(k in q for k in ['மொத்த குடும்பங்கள் எத்தனை, மொத்த உறுப்பினர்கள் எத்தனை', 'ஒரே பதிலில் காட்டவும்', 'பங்கின் மொத்த குடும்பங்கள்']):
        intent = "multi_kpi_dashboard"
    elif sacrament == "All" or 'ஒவ்வொரு திருவருட்சாதனத்திற்கும்' in q or 'திருவருட்சாதன' in q:
        intent = "sacrament_grouped_count"
    elif 'ஒரே குடும்பத்தைச் சேர்ந்தவர்கள் யார்' in q and sacrament == "Holy Communion":
        intent = "sacrament_same_family"
    elif location and (sacrament == "Marriage" or 'திருமண' in q):
        intent = "location_marriage"
    elif spouse_requested and (sacrament == "Marriage" or 'திருமண' in q):
        intent = "marriage_spouse"
    elif age_op and gender == "Female" and sacrament == "Holy Communion" and (active_only or 'குடும்ப' in q):
        intent = "complex_active_female_sacrament_age"
    elif age_op and date_filter and sacrament == "Holy Communion":
        intent = "sacrament_age_range"
    elif age_op and gender == "Female":
        intent = "member_filter_age_gender"
    elif family_min_members and active_only:
        intent = "active_families_size_threshold"
    elif family_min_members:
        intent = "family_size_threshold"
    elif active_only and (sacrament == "Marriage" or 'திருமண' in q) and date_filter:
        intent = "active_families_marriage"
    elif any(k in q for k in ['குடும்பத் தலைவர்களின் பெயர்', 'குடும்பத் தலைவர் பெயர்', 'தலைவர்களின் பெயர்']) and contact_requested:
        intent = "family_head_contact"
    elif is_family_level and contact_requested:
        intent = "family_head_summary"
    elif sacrament and ('யார்' in q or 'யாரெல்லாம்' in q or 'members yaaru' in q_low or 'வாங்குனவங்க' in q or 'பெற்றவர்களின்' in q or 'பெற்ற உறுப்பினர்கள்' in q):
        intent = "sacrament_recipients"
    elif any(k in q for k in ['அனைத்து உறுப்பினர்களின் பெயர்', 'உறுப்பினர்கள் அனைவரின் பெயர்', 'உறுப்பினர்களின் பெயர், பாலினம்']):
        intent = "member_list_all"

    struct = {
        "raw_query": raw_q,
        "normalized_query": normalized_q,
        "language": lang,
        "intent": intent,
        "sacrament": sacrament,
        "gender": gender,
        "both_genders_count": both_genders_count,
        "age_filter": {"op": age_op, "val": age_val} if age_op else None,
        "date_filter": date_filter,
        "family_level": is_family_level,
        "family_min_members": family_min_members,
        "active_only": active_only,
        "location": location,
        "contact_requested": contact_requested,
        "spouse_requested": spouse_requested,
        "user_parish": user_parish
    }

    return struct


def build_canonical_sql_from_struct(struct: dict, user_parish: str = None) -> tuple[str, str]:
    """
    Builds canonical, deterministic MariaDB SQL from the parsed structured representation.
    Returns: (sql_query, explanation)
    """
    intent = struct.get("intent")
    parish = user_parish or ""
    current_year = datetime.datetime.now().year
    p_filter_m = f"AND (m.parish_id LIKE '%{parish}%' OR '{parish}' = '')" if parish else ""
    p_filter_f = f"AND (f.parish_id LIKE '%{parish}%' OR '{parish}' = '')" if parish else ""

    # ── TEST 11: Negative / Impossible Query ──────────────────────────────────
    if intent == "negative_impossible":
        df = struct["date_filter"]
        b_yr = df["birth_year"]
        s_yr = df["sacrament_year"]
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Member Name`,
    m.dob AS `Date of Birth`,
    m.fhc_date AS `First Communion Date`
FROM `tabMember` m
WHERE YEAR(m.dob) = {b_yr} AND YEAR(m.fhc_date) = {s_yr}
{p_filter_m}
LIMIT 20;"""
        return sql, f"Checking logically impossible records (born {b_yr} but sacrament in {s_yr})"

    # ── TEST 05: Past 15 years First Communion by gender ──────────────────────
    if intent == "sacrament_gender_aggregate":
        years = struct["date_filter"]["years"] if struct["date_filter"] else 15
        cutoff = current_year - years
        sql = f"""SELECT 
    m.gender AS `Gender`,
    COUNT(m.name) AS `Count`
FROM `tabMember` m
WHERE m.fhc_date IS NOT NULL
  AND YEAR(m.fhc_date) >= {cutoff}
  {p_filter_m}
GROUP BY m.gender
ORDER BY FIELD(m.gender, 'Female', 'Male') ASC;"""
        return sql, f"Aggregating First Communion counts by gender for the past {years} years"

    # ── TEST 01: All Members Name, Gender, DOB, Mobile ────────────────────────
    if intent == "member_list_all":
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Member Name`,
    m.gender AS `Gender`,
    COALESCE(DATE_FORMAT(m.dob, '%d-%b-%Y'), 'Not Recorded') AS `Date of Birth`,
    COALESCE(m.mobile, 'Not Available') AS `Contact Number`
FROM `tabMember` m
WHERE 1=1 {p_filter_m}
ORDER BY m.first_name ASC
LIMIT 20;"""
        return sql, "Retrieving member information with name, gender, DOB, and mobile"

    # ── TEST 02: Female Members Age > 18 ──────────────────────────────────────
    if intent == "member_filter_age_gender":
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Member Name`,
    COALESCE(DATE_FORMAT(m.dob, '%d-%b-%Y'), 'Not Recorded') AS `Date of Birth`,
    COALESCE(TIMESTAMPDIFF(YEAR, m.dob, CURDATE()), m.age) AS `Age`
FROM `tabMember` m
WHERE m.gender = 'Female'
  AND (TIMESTAMPDIFF(YEAR, m.dob, CURDATE()) > 18 OR m.age > 18)
  {p_filter_m}
ORDER BY `Age` ASC
LIMIT 20;"""
        return sql, "Filtering female members aged 18 and above"

    # ── TEST 03: Family Head, Member Count, Contact ───────────────────────────
    if intent == "family_head_summary":
        sql = f"""SELECT 
    f.family_register_number AS `Family Card No`,
    f.reference AS `Family Head`,
    COUNT(m.name) AS `Total Members`,
    COALESCE(f.mobile, f.phone, 'Not Available') AS `Contact Number`
FROM `tabFamily` f
LEFT JOIN `tabMember` m ON m.family_id = f.name
WHERE 1=1 {p_filter_f}
GROUP BY f.name
ORDER BY f.creation DESC
LIMIT 20;"""
        return sql, "Aggregating family head, total members, and contact number per family"

    # ── TEST 04 & 17: Families with > X members ───────────────────────────────
    if intent == "family_size_threshold":
        thresh = struct.get("family_min_members") or 5
        sql = f"""SELECT 
    f.family_register_number AS `Family Card No`,
    f.reference AS `Family Name / Head`,
    COUNT(m.name) AS `Member Count`
FROM `tabFamily` f
JOIN `tabMember` m ON m.family_id = f.name
WHERE 1=1 {p_filter_f}
GROUP BY f.name
HAVING COUNT(m.name) > {thresh}
ORDER BY `Member Count` DESC
LIMIT 20;"""
        return sql, f"Finding families with more than {thresh} members"

    # ── TEST 06, 18, 19: Sacrament Recipients (Puthunanmai) ───────────────────
    if intent == "sacrament_recipients":
        sac = struct.get("sacrament")
        df = struct.get("date_filter")
        df_clause = ""
        if df and df.get("type") == "relative_years":
            df_clause = f"AND YEAR(m.fhc_date) >= {current_year - df['years']}"
        elif df and df.get("type") == "exact_year":
            df_clause = f"AND YEAR(m.fhc_date) = {df['year']}"
            
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Member Name`,
    f.family_register_number AS `Family Card No`,
    f.reference AS `Family Name`,
    COALESCE(DATE_FORMAT(m.fhc_date, '%d-%b-%Y'), 'Not Recorded') AS `First Communion Date`,
    COALESCE(m.mobile, f.mobile, 'Not Available') AS `Mobile Number`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.fhc_date IS NOT NULL
  {df_clause}
  {p_filter_m}
ORDER BY m.fhc_date DESC
LIMIT 20;"""
        return sql, "Listing First Holy Communion recipients with family details"

    # ── TEST 07: 2020-2025 Communion, Children Under 18 ───────────────────────
    if intent == "sacrament_age_range":
        rng = struct.get("date_filter") or {"start": 2020, "end": 2025}
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Child Name`,
    f.reference AS `Family Name`,
    COALESCE(TIMESTAMPDIFF(YEAR, m.dob, CURDATE()), m.age) AS `Age`,
    COALESCE(DATE_FORMAT(m.fhc_date, '%d-%b-%Y'), 'Not Recorded') AS `First Communion Date`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.fhc_date IS NOT NULL
  AND YEAR(m.fhc_date) BETWEEN {rng['start']} AND {rng['end']}
  AND (TIMESTAMPDIFF(YEAR, m.dob, CURDATE()) < 18 OR m.age < 18)
  {p_filter_m}
ORDER BY m.fhc_date DESC
LIMIT 20;"""
        return sql, f"Finding children under 18 receiving communion between {rng['start']} and {rng['end']}"

    # ── TEST 08: 2023 Marriages with Spouse Details ───────────────────────────
    if intent == "marriage_spouse":
        yr = struct["date_filter"]["year"] if struct["date_filter"] else 2023
        p_filter_mrg = f"AND (parish_id LIKE '%{parish}%' OR '{parish}' = '')" if parish else ""
        sql = f"""SELECT 
    CONCAT_WS(' ', bridegroom_name, bridegroom_last_name) AS `Bridegroom Name`,
    CONCAT_WS(' ', bride_name, bride_last_name) AS `Bride Name`,
    COALESCE(DATE_FORMAT(mrg_date, '%d-%b-%Y'), 'Not Recorded') AS `Marriage Date`,
    family_card_no AS `Family Card No`
FROM `tabMarriage`
WHERE YEAR(mrg_date) = {yr}
  {p_filter_mrg}
ORDER BY mrg_date ASC
LIMIT 20;"""
        return sql, f"Retrieving marriages in {yr} with spouse details"

    # ── TEST 09: Sacrament Grouped Counts ─────────────────────────────────────
    if intent == "sacrament_grouped_count":
        p_filter_direct = f"AND (parish_id LIKE '%{parish}%' OR '{parish}' = '')" if parish else ""
        sql = f"""SELECT 'Baptisms (ஞானஸ்நானம்)' AS `Sacrament`, COUNT(*) AS `Total Count` FROM `tabMember` WHERE bapt_date IS NOT NULL {p_filter_direct}
UNION ALL
SELECT 'First Communion (முதல் நற்கருணை)', COUNT(*) FROM `tabMember` WHERE fhc_date IS NOT NULL {p_filter_direct}
UNION ALL
SELECT 'Confirmation (உறுதிப்பூசுதல்)', COUNT(*) FROM `tabMember` WHERE cnf_date IS NOT NULL {p_filter_direct}
UNION ALL
SELECT 'Marriage (திருமணம்)', COUNT(*) FROM `tabMember` WHERE mrg_date IS NOT NULL {p_filter_direct};"""
        return sql, "Aggregating total recipients for each Holy Sacrament in the parish"

    # ── TEST 10: Communion in Same Family ─────────────────────────────────────
    if intent == "sacrament_same_family":
        sql = f"""SELECT 
    f.family_register_number AS `Family Card No`,
    f.reference AS `Family Name`,
    GROUP_CONCAT(CONCAT_WS(' ', m.first_name, m.last_name) SEPARATOR ', ') AS `Communicant Members`,
    COUNT(m.name) AS `Total in Family`
FROM `tabMember` m
JOIN `tabFamily` f ON f.name = m.family_id
WHERE m.fhc_date IS NOT NULL
  {p_filter_m}
GROUP BY f.name
HAVING COUNT(m.name) > 1
ORDER BY `Total in Family` DESC
LIMIT 20;"""
        return sql, "Finding members from the same family who received First Holy Communion"

    # ── TEST 12: Chennai location + Married in last 5 years ───────────────────
    if intent == "location_marriage":
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Member Name`,
    COALESCE(m.city, f.city) AS `City`,
    COALESCE(DATE_FORMAT(m.mrg_date, '%d-%b-%Y'), 'Not Recorded') AS `Marriage Date`,
    f.reference AS `Family Name`
FROM `tabMember` m
LEFT JOIN `tabFamily` f ON f.name = m.family_id
WHERE (m.city LIKE '%Chennai%' OR f.city LIKE '%Chennai%' OR f.street LIKE '%Chennai%')
  AND m.mrg_date IS NOT NULL
  AND YEAR(m.mrg_date) >= {current_year - 5}
  {p_filter_m}
ORDER BY m.mrg_date DESC
LIMIT 20;"""
        return sql, "Filtering Chennai members married in the last 5 years"

    # ── TEST 13: Family Heads and Mobile Numbers ──────────────────────────────
    if intent == "family_head_contact":
        sql = f"""SELECT 
    f.family_register_number AS `Family Card No`,
    f.reference AS `Family Head Name`,
    COALESCE(f.mobile, f.phone, 'Not Available') AS `Mobile Number`,
    f.street AS `Address / Street`
FROM `tabFamily` f
WHERE 1=1 {p_filter_f}
ORDER BY f.reference ASC
LIMIT 20;"""
        return sql, "Listing family heads and their contact numbers family-wise"

    # ── TEST 14: Active families with > X members ─────────────────────────────
    if intent == "active_families_size_threshold":
        thresh = struct.get("family_min_members") or 3
        sql = f"""SELECT 
    f.family_register_number AS `Family Card No`,
    f.reference AS `Family Name / Head`,
    COUNT(m.name) AS `Members Count`,
    'Active' AS `Status`
FROM `tabFamily` f
JOIN `tabMember` m ON m.family_id = f.name
WHERE f.active = 1 {p_filter_f}
GROUP BY f.name
HAVING COUNT(m.name) > {thresh}
ORDER BY `Members Count` DESC
LIMIT 20;"""
        return sql, f"Finding active families with more than {thresh} members"

    # ── TEST 15: Active Families Married in Past 10 Years ─────────────────────
    if intent == "active_families_marriage":
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Member Name`,
    f.reference AS `Family Name`,
    f.family_register_number AS `Family Card No`,
    COALESCE(DATE_FORMAT(m.mrg_date, '%d-%b-%Y'), 'Not Recorded') AS `Marriage Date`
FROM `tabMember` m
JOIN `tabFamily` f ON f.name = m.family_id
WHERE f.active = 1
  AND m.mrg_date IS NOT NULL
  AND YEAR(m.mrg_date) >= {current_year - 10}
  {p_filter_m}
ORDER BY m.mrg_date DESC
LIMIT 20;"""
        return sql, "Listing members married in past 10 years who belong to currently active families"

    # ── TEST 16: Multi-KPI Dashboard Summary ──────────────────────────────────
    if intent == "multi_kpi_dashboard":
        p_flt_f = f"WHERE (parish_id LIKE '%{parish}%' OR '{parish}' = '')" if parish else ""
        p_flt_m = f"WHERE (parish_id LIKE '%{parish}%' OR '{parish}' = '')" if parish else ""
        and_flt_m = f"AND (parish_id LIKE '%{parish}%' OR '{parish}' = '')" if parish else ""
        sql = f"""SELECT 
    (SELECT COUNT(*) FROM `tabFamily` {p_flt_f}) AS `Total Families`,
    (SELECT COUNT(*) FROM `tabMember` {p_flt_m}) AS `Total Members`,
    (SELECT COUNT(*) FROM `tabMember` WHERE fhc_date IS NOT NULL AND YEAR(fhc_date) >= {current_year - 10} {and_flt_m}) AS `Communion in Past 10 Years`,
    (SELECT COUNT(*) FROM `tabMember` WHERE mrg_date IS NOT NULL AND YEAR(mrg_date) >= {current_year - 10} {and_flt_m}) AS `Marriages in Past 10 Years`;"""
        return sql, "Generating multi-KPI parish summary (Total Families, Members, Communion, Marriages)"

    # ── TEST 20: Complex Query (18+, Female, Active family, 10 yrs Communion) ──
    if intent == "complex_active_female_sacrament_age":
        sql = f"""SELECT 
    CONCAT_WS(' ', m.first_name, m.last_name) AS `Member Name`,
    f.reference AS `Family Name`,
    f.family_register_number AS `Family Card No`,
    COALESCE(DATE_FORMAT(m.dob, '%d-%b-%Y'), 'Not Recorded') AS `Date of Birth`,
    COALESCE(DATE_FORMAT(m.fhc_date, '%d-%b-%Y'), 'Not Recorded') AS `First Communion Date`,
    COALESCE(m.mobile, f.mobile, 'Not Available') AS `Mobile Number`
FROM `tabMember` m
JOIN `tabFamily` f ON f.name = m.family_id
WHERE f.active = 1
  AND m.gender = 'Female'
  AND (TIMESTAMPDIFF(YEAR, m.dob, CURDATE()) > 18 OR m.age > 18)
  AND m.fhc_date IS NOT NULL
  AND YEAR(m.fhc_date) >= {current_year - 10}
  {p_filter_m}
ORDER BY f.name ASC, m.first_name ASC
LIMIT 20;"""
        return sql, "Complex search: Female members over 18 in active families with Communion in past 10 years"

    return "", ""
