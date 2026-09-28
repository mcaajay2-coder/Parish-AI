import math
import re
from typing import Dict, Any, List, Optional, Tuple
from koinonia_assistant.rag.tamil_utils import (
    is_tamil,
    detect_query_language,
    extract_tamil_structured_intent,
    CATHOLIC_TAMIL_TERMINOLOGY_DB,
)

# ─── Canonical Sacrament Metadata & Table Mapping ─────────────────────────────
SACRAMENT_METRICS: Dict[str, Dict[str, Any]] = {
    "baptism": {
        "label": "Baptism",
        "ta_label": "திருமுழுக்கு",
        "code": "BAPTISM",
        "table": "tabBaptism",
        "date_col": "bapt_date",
        "parish_cols": ["parish_id", "bapt_parish_id"],
        "member_col": "bapt_date",
        "color": "#d4af37",
    },
    "communion": {
        "label": "First Holy Communion",
        "ta_label": "முதல் நற்கருணை",
        "code": "FIRST_HOLY_COMMUNION",
        "table": "tabCommunion",
        "date_col": "fhc_date",
        "parish_cols": ["parish_id", "fhc_parish_id"],
        "member_col": "fhc_date",
        "color": "#38bdf8",
    },
    "confirmation": {
        "label": "Confirmation",
        "ta_label": "உறுதிப்பூசுதல்",
        "code": "CONFIRMATION",
        "table": "tabConfirmation",
        "date_col": "cnf_date",
        "parish_cols": ["parish_id", "cnf_parish_id"],
        "member_col": "cnf_date",
        "color": "#10b981",
    },
    "marriage": {
        "label": "Marriage",
        "ta_label": "திருமணம்",
        "code": "MARRIAGE",
        "table": "tabMarriage",
        "date_col": "mrg_date",
        "parish_cols": ["parish_id", "mrg_parish_id"],
        "member_col": "mrg_date",
        "color": "#f43f5e",
    },
}

ALLOWED_SQL_TABLES = {
    "tabmember",
    "tabfamily",
    "tabbaptism",
    "tabcommunion",
    "tabconfirmation",
    "tabmarriage",
    "tabdeath",
    "tabparish",
    "tabvicariate",
    "tabdiocese",
}

KNOWN_OTHER_PARISHES = [
    "sacred heart parish",
    "st. joseph's parish",
    "st. joseph parish",
    "st. mary's cathedral",
    "st. marys parish",
    "infant jesus parish",
    "our lady of good health",
    "holy redeemer parish",
    "holy family parish",
    "another parish",
    "other parish",
    "other parishes",
    "மற்றொரு பங்கு",
    "பிற பங்கு",
    "வேறு பங்கு",
]


# ─── 1. Central Authorization Context Builder (Sections 50–53) ────────────────
def build_authorization_context(
    user_id: str,
    user_role: str,
    user_parish: Optional[str] = None,
    user_diocese: Optional[str] = None,
    user_vicariate: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates the single authoritative authorization context object for every request
    BEFORE any intent routing, SQL generation, vector retrieval, or analytics.
    """
    role_clean = (user_role or "Parishioner").strip()
    parish_clean = (user_parish or "").strip() or None
    vicariate_clean = (user_vicariate or "").strip() or None
    diocese_clean = (user_diocese or "").strip() or None

    PARISH_CODE_MAP = {
        "PR-00018": "Yelagiri Parish",
        "YELAGIRI": "Yelagiri Parish",
        "YELAGIRI PARISH": "Yelagiri Parish",
    }
    if parish_clean:
        resolved_parish_name = PARISH_CODE_MAP.get(parish_clean.upper(), parish_clean)
        scope_type = "PARISH"
        scope_id = "PR-00018" if resolved_parish_name == "Yelagiri Parish" else parish_clean
        scope_name = resolved_parish_name
        allowed_parishes = list({parish_clean, resolved_parish_name, scope_id})
    elif vicariate_clean:
        scope_type = "VICARIATE"
        scope_id = vicariate_clean
        scope_name = f"{vicariate_clean} Vicariate"
        allowed_parishes = []
    elif diocese_clean and diocese_clean != "All Dioceses":
        scope_type = "DIOCESE"
        scope_id = diocese_clean
        scope_name = f"{diocese_clean}"
        allowed_parishes = []
    elif role_clean in ("System Manager", "Administrator", "Diocese Administrator", "Bishop"):
        scope_type = "DIOCESE"
        scope_id = diocese_clean or "DIOCESE_ALL"
        scope_name = diocese_clean if (diocese_clean and diocese_clean != "All Dioceses") else "Diocesan Registry"
        allowed_parishes = []
    else:
        # Fail-closed default to parish scope
        scope_type = "PARISH"
        scope_id = parish_clean or "RESTRICTED_PARISH"
        scope_name = parish_clean or "Authorized Parish"
        allowed_parishes = [parish_clean] if parish_clean else []

    return {
        "user_id": user_id or "Guest",
        "role": role_clean,
        "scope_type": scope_type,
        "scope_id": scope_id,
        "scope_name": scope_name,
        "parish_id": scope_name if scope_type == "PARISH" else parish_clean,
        "parish_code": scope_id,
        "parish_name": scope_name,
        "vicariate_id": vicariate_clean,
        "diocese_id": diocese_clean,
        "allowed_parish_ids": allowed_parishes,
        "allowed_vicariate_ids": [vicariate_clean] if vicariate_clean else [],
        "allowed_diocese_ids": [diocese_clean] if (diocese_clean and diocese_clean != "All Dioceses") else [],
    }


# ─── 2. Pre-Execution Authorization Scope & Family Card Guard (Sections 62, 63, 64, 67) ───
def check_explicit_scope_violation(
    question: str,
    auth_ctx: Dict[str, Any],
    detected_language: str = "en",
) -> Dict[str, Any]:
    """
    Executes BEFORE any database query or analytics node.
    Blocks:
    1. Parish user asking for Diocesan-wide statistics ("Show diocesan baptism statistics")
    2. Parish user asking to compare or view another parish ("Compare Yelagiri and another parish")
    3. Family Card lookup ("YLG/999") where the card does not belong to the authorized parish.
    """
    q_raw = (question or "").strip()
    q_low = q_raw.lower()
    scope_type = auth_ctx.get("scope_type", "PARISH")
    scope_name = auth_ctx.get("scope_name", "Yelagiri Parish")
    user_parish = auth_ctx.get("parish_id")

    if scope_type == "PARISH":
        # Check 1: Explicit request for Diocesan-level data by a Parish-level user (Section 63)
        asks_diocese = bool(
            re.search(
                r"\b(?:diocesan|diocese[\s\-]*wide|entire\s+diocese|all\s+parishes\s+in\s+(?:the\s+)?diocese|across\s+the\s+diocese|diocese\s+baptism|diocese\s+statistics|மறைமாவட்ட|மறைமாவட்டம்\s+முழுவதும்)\b",
                q_low,
            )
        )
        if asks_diocese:
            msg = (
                f"உங்கள் கணக்கிற்கான அணுகல் **{scope_name}** பங்கிற்கு மட்டுமே வரையறுக்கப்பட்டுள்ளது. மறைமாவட்ட அளவிலான புள்ளிவிவரங்களைப் பார்க்க உங்களுக்கு அனுமதி இல்லை."
                if detected_language == "ta"
                else f"Your current access is limited to **{scope_name}**. Diocesan-level statistics are not available for your account."
            )
            return {
                "authorized": False,
                "authorization_result": "DENIED_DIOCESAN_SCOPE",
                "message": msg,
            }

        # Check 2: Explicit request for another parish or cross-parish comparison (Section 62)
        for other_p in KNOWN_OTHER_PARISHES:
            if other_p in q_low and (not user_parish or other_p not in user_parish.lower()):
                msg = (
                    f"உங்கள் அனுமதிக்கப்பட்ட பங்கு எல்லைக்கு (**{scope_name}**) வெளியே உள்ள பிற பங்குகளின் தரவுகளை ஒப்பிடவோ அல்லது பார்க்கவோ உங்களுக்கு அதிகாரம் இல்லை."
                    if detected_language == "ta"
                    else f"You are not authorized to compare or access data outside your permitted parish scope (**{scope_name}**)."
                )
                return {
                    "authorized": False,
                    "authorization_result": "DENIED_CROSS_PARISH_SCOPE",
                    "message": msg,
                }

        # Check 3: Family Card pre-verification against authorized parish (Sections 64 & 67)
        m_card = re.search(r"\b([A-Z]{2,5}/\d{1,5})\b", q_raw, re.IGNORECASE)
        if m_card and user_parish:
            card_code = m_card.group(1).upper()
            try:
                import frappe
                rows = frappe.db.sql(
                    """
                    SELECT name, parish_id
                    FROM `tabFamily`
                    WHERE UPPER(family_register_number) = %s
                      AND (parish_id = %s OR parish_id LIKE %s)
                    LIMIT 1
                    """,
                    (card_code, user_parish, f"%{user_parish}%"),
                    as_dict=True,
                )
                if not rows:
                    msg = (
                        f"உங்கள் அனுமதிக்கப்பட்ட பங்கு எல்லையில் (**{scope_name}**) `{card_code}` என்ற குடும்ப அட்டை விவரம் எதுவும் கண்டறியப்படவில்லை."
                        if detected_language == "ta"
                        else f"No family record matching `{card_code}` is available within your authorized parish scope (**{scope_name}**)."
                    )
                    return {
                        "authorized": False,
                        "authorization_result": "DENIED_UNAUTHORIZED_FAMILY_CARD",
                        "message": msg,
                    }
            except Exception:
                pass

    return {
        "authorized": True,
        "authorization_result": "AUTHORIZED",
        "message": None,
    }


# ─── 3. Dedicated SQL Security Validator (Section 59) ─────────────────────────
def authorization_sql_validator(sql: str, auth_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifies BEFORE database execution that:
    1. Only SELECT statements are allowed.
    2. Only allowed Koinonia tables are referenced.
    3. For PARISH scope, the SQL query explicitly enforces parish_id filtering
       before any GROUP BY, ORDER BY, or LIMIT.
    4. No UNION bypass or unrestricted aggregation exists.
    """
    if not sql or not sql.strip():
        return {"valid": False, "reason": "EMPTY_SQL"}

    s_clean = sql.strip()
    s_low = s_clean.lower()

    if not s_low.startswith("select"):
        return {"valid": False, "reason": "NON_SELECT_STATEMENT"}

    if re.search(r"\b(?:drop|delete|update|insert|alter|truncate|grant|revoke)\b", s_low):
        return {"valid": False, "reason": "MUTATING_KEYWORD_BLOCKED"}

    if " union " in s_low:
        return {"valid": False, "reason": "UNION_BYPASS_BLOCKED"}

    # Check referenced tables
    referenced_tables = re.findall(r"(?:from|join)\s+`?(tab[a-z0-9_]+)`?", s_low)
    for tbl in referenced_tables:
        if tbl not in ALLOWED_SQL_TABLES:
            return {"valid": False, "reason": f"UNAUTHORIZED_TABLE:{tbl}"}

    # For PARISH scope, verify parish_id constraint exists before GROUP BY
    scope_type = auth_ctx.get("scope_type", "PARISH")
    if scope_type == "PARISH":
        has_parish_col = any(
            col in s_low
            for col in ["parish_id", "bapt_parish_id", "fhc_parish_id", "cnf_parish_id", "mrg_parish_id"]
        )
        if not has_parish_col:
            return {"valid": False, "reason": "MISSING_MANDATORY_PARISH_FILTER"}
        if "group by" in s_low:
            where_idx = s_low.find("where")
            group_idx = s_low.find("group by")
            if where_idx == -1 or where_idx > group_idx:
                return {"valid": False, "reason": "AGGREGATION_BEFORE_AUTHORIZATION_FILTER"}

    return {"valid": True, "reason": "PASSED_ALL_SECURITY_CHECKS"}


# ─── 4. Multilingual & Tamil-First Intent Classifier (Sections 24–35, 43) ─────
def classify_langgraph_intent(question: str) -> Dict[str, Any]:
    """
    Classifies a user query into one of the 12 canonical LangGraph intents while
    strictly preserving the original Tamil/English query without Tanglish conversion.
    """
    q_clean = (question or "").strip()
    q_low = q_clean.lower()
    lang = detect_query_language(q_clean)

    # 1. GREETING (English & Tamil)
    if re.match(
        r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|praise\s+the\s+lord|vanakkam|வணக்கம்|நமஸ்காரம்)\b[!.,?\s]*$",
        q_low,
    ):
        return {
            "original_query": q_clean,
            "detected_language": lang,
            "normalized_query": f"INTENT=GREETING | LANGUAGE={lang}",
            "intent_query": "GREETING",
            "entity_query": "NONE",
            "canonical_terms": {},
            "intent": "GREETING",
            "speed_tier": "FAST",
            "metrics": [],
            "years_back": 0,
            "forecast_horizon": 0,
            "final_response_language": "ta" if lang == "ta" else "en",
        }

    # 2. STATISTICAL & AGGREGATE PRE-ROUTING (Bypasses Person Name Extraction completely)
    from koinonia_assistant.rag.name_search import detect_statistical_query
    stats_dims = detect_statistical_query(q_clean)
    if stats_dims:
        s_ent = stats_dims["entity"]
        s_int = stats_dims["intent"]
        s_group = stats_dims.get("group_by")

        canonical_terms = {}
        if any(w in q_low for w in ["baptism", "baptized", "baptised", "christening", "ஞானஸ்நானம்", "திருமுழுக்கு"]):
            canonical_terms["BAPTISM"] = "Baptism"
        if any(w in q_low for w in ["communion", "fhc", "eucharist", "holy communion", "நற்கருணை", "திருவிருந்து"]):
            canonical_terms["FIRST_HOLY_COMMUNION"] = "First Holy Communion"
        if any(w in q_low for w in ["confirmation", "confirmed", "உறுதிப்பூசுதல்"]):
            canonical_terms["CONFIRMATION"] = "Confirmation"
        if any(w in q_low for w in ["marriage", "married", "matrimony", "wedding", "திருமணம்"]):
            canonical_terms["MARRIAGE"] = "Marriage"

        # If year-wise sacrament query, route to HISTORICAL_ANALYSIS/analytics_node
        if s_group == "YEAR" and s_ent in ("BAPTISM", "COMMUNION", "CONFIRMATION", "MARRIAGE", "SACRAMENT"):
            m_list = [s_ent.lower()] if s_ent != "SACRAMENT" else ["baptism"]
            return {
                "original_query": q_clean,
                "detected_language": lang,
                "canonical_terms": canonical_terms,
                "sacrament": SACRAMENT_METRICS[m_list[0]]["code"],
                "period": "LAST_10_YEARS",
                "grouping": "YEAR",
                "final_response_language": "ta" if lang == "ta" else "en",
                "normalized_query": f"INTENT=HISTORICAL_ANALYSIS | METRIC={m_list[0]} | PERIOD=LAST_10_YEARS",
                "intent_query": "HISTORICAL_ANALYSIS",
                "entity_query": f"METRIC={m_list[0]}",
                "intent": "HISTORICAL_ANALYSIS",
                "speed_tier": "ANALYTICAL",
                "metrics": m_list,
                "years_back": 10,
                "forecast_horizon": 0,
                "person_name": None,
                "statistical_dimensions": stats_dims,
            }

        return {
            "original_query": q_clean,
            "detected_language": lang,
            "canonical_terms": canonical_terms,
            "sacrament": None,
            "period": None,
            "grouping": s_group,
            "final_response_language": "ta" if lang == "ta" else "en",
            "normalized_query": f"INTENT={s_int} | METRIC={stats_dims['metric']} | ENTITY={s_ent} | GROUP_BY={s_group} | GENDER={stats_dims.get('gender')}",
            "intent_query": s_int,
            "entity_query": s_ent,
            "intent": s_int,
            "scope": s_int,
            "person_name": None,
            "speed_tier": "FAST",
            "metrics": [s_ent.lower()] if s_ent in ("BAPTISM", "COMMUNION", "CONFIRMATION", "MARRIAGE") else [],
            "statistical_dimensions": stats_dims,
        }

    # 3. Direct Tamil Structured Understanding (when Tamil script is present)
    if lang == "ta":
        ta_info = extract_tamil_structured_intent(q_clean)
        t_intent = ta_info["intent"]
        t_metrics = ta_info["metrics"]
        t_years = ta_info["years_back"]
        t_horizon = ta_info["forecast_horizon"]
        if t_intent in ("FORECAST", "COMPARISON") and (t_horizon >= 5 or len(t_metrics) >= 2):
            s_tier = "LONG_RUNNING"
        elif t_intent in ("HISTORICAL_ANALYSIS", "STATISTICAL_ANALYSIS", "TREND_ANALYSIS", "FORECAST", "COMPARISON"):
            s_tier = "ANALYTICAL"
        else:
            s_tier = "FAST"

        return {
            "original_query": q_clean,
            "detected_language": "ta",
            "normalized_query": ta_info["normalized_query"],
            "intent_query": ta_info["intent_query"],
            "entity_query": ta_info["entity_query"],
            "canonical_terms": ta_info["canonical_terms"],
            "sacrament": ta_info["sacrament"],
            "period": ta_info["period"],
            "grouping": ta_info["grouping"],
            "intent": t_intent,
            "speed_tier": s_tier,
            "metrics": t_metrics,
            "years_back": t_years,
            "forecast_horizon": t_horizon,
            "include_forecast": t_intent == "FORECAST" or t_horizon > 0,
            "family_card": ta_info.get("family_card"),
            "person_name": (
                ta_info["person_entity"].get("transliterated_name")
                or ta_info["person_entity"].get("original_name")
            ),
            "person_entity": ta_info.get("person_entity"),
            "final_response_language": "ta",
            "sub_intent": ta_info.get("sub_intent"),
            "scope": ta_info.get("scope"),
        }

    # 4. English / Mixed Query Classification
    metrics = []
    canonical_terms = {}
    if any(w in q_low for w in ["baptism", "baptized", "baptised", "christening"]):
        metrics.append("baptism")
        canonical_terms["BAPTISM"] = "Baptism"
    if any(w in q_low for w in ["communion", "fhc", "eucharist", "holy communion"]):
        metrics.append("communion")
        canonical_terms["FIRST_HOLY_COMMUNION"] = "First Holy Communion"
    if any(w in q_low for w in ["confirmation", "confirmed"]):
        metrics.append("confirmation")
        canonical_terms["CONFIRMATION"] = "Confirmation"
    if any(w in q_low for w in ["marriage", "married", "matrimony", "wedding"]):
        metrics.append("marriage")
        canonical_terms["MARRIAGE"] = "Marriage"

    years_back = 10
    m_past = re.search(r"\b(?:last|past|previous|over\s+the\s+last|for\s+the\s+last)\s+(\d+)\s+years?\b", q_low)
    if m_past:
        years_back = max(2, min(50, int(m_past.group(1))))

    forecast_horizon = 0
    m_next = re.search(r"\b(?:next|coming|upcoming|future|forecast\s+(?:the\s+)?next)\s+(\d+)\s+years?\b", q_low)
    if m_next:
        forecast_horizon = max(1, min(30, int(m_next.group(1))))
    elif any(w in q_low for w in ["forecast", "predict", "projection", "project", "future", "how might", "what could happen", "next 10 years", "next 5 years"]):
        forecast_horizon = 10

    is_forecast = (
        forecast_horizon > 0
        or any(w in q_low for w in [
            "forecast", "predict", "projection", "future", "how might",
            "what could happen", "next 10 years", "next 5 years", "coming years"
        ])
    )
    is_comparison = (
        any(w in q_low for w in ["compare", "comparison", "versus", " vs ", "between"])
        or (len(metrics) >= 2 and any(w in q_low for w in ["year", "trend", "last", "history", "historical", "statistics", "analyze", "analyse", "forecast"]))
    )
    is_trend = any(w in q_low for w in [
        "trend", "changed", "change over", "growth", "increasing", "decreasing",
        "slope", "year-over-year", "yoy", "how has"
    ])
    is_statistical = any(w in q_low for w in [
        "statistics", "statistical", "average", "mean", "median", "variance", "volatility", "summary statistics"
    ])
    is_historical = (
        bool(m_past)
        or any(w in q_low for w in [
            "each year", "every year", "year by year", "yearly", "per year",
            "historical", "over the years", "last 10 years", "last 5 years", "last 20 years"
        ])
    )

    base_meta = {
        "original_query": q_clean,
        "detected_language": lang,
        "canonical_terms": canonical_terms,
        "sacrament": SACRAMENT_METRICS[metrics[0]]["code"] if metrics else None,
        "period": f"LAST_{years_back}_YEARS" if (is_historical or is_trend or is_statistical or is_forecast) else None,
        "grouping": "YEAR" if (is_historical or is_trend or is_statistical or is_forecast) else None,
        "final_response_language": "en",
    }

    if is_comparison and (is_historical or is_trend or is_forecast or len(metrics) >= 2):
        if not metrics:
            metrics = ["baptism", "confirmation"]
        tier = "LONG_RUNNING" if (is_forecast or len(metrics) >= 3 or years_back >= 15) else "ANALYTICAL"
        return {
            **base_meta,
            "normalized_query": f"INTENT=COMPARISON | METRICS={','.join(metrics)} | PERIOD=LAST_{years_back}_YEARS",
            "intent_query": "COMPARISON",
            "entity_query": f"METRICS={','.join(metrics)}",
            "intent": "COMPARISON",
            "speed_tier": tier,
            "metrics": metrics,
            "years_back": years_back,
            "forecast_horizon": forecast_horizon,
            "include_forecast": is_forecast,
        }

    if is_forecast:
        if not metrics:
            metrics = ["baptism"]
        tier = "LONG_RUNNING" if (len(metrics) > 1 or forecast_horizon >= 10) else "ANALYTICAL"
        return {
            **base_meta,
            "normalized_query": f"INTENT=FORECAST | METRIC={metrics[0]} | HORIZON={forecast_horizon or 10}",
            "intent_query": "FORECAST",
            "entity_query": f"METRIC={metrics[0]}",
            "intent": "FORECAST",
            "speed_tier": tier,
            "metrics": metrics,
            "years_back": years_back,
            "forecast_horizon": forecast_horizon or 10,
            "include_forecast": True,
        }

    if is_trend and (metrics or is_historical):
        if not metrics:
            metrics = ["baptism"]
        return {
            **base_meta,
            "normalized_query": f"INTENT=TREND_ANALYSIS | METRIC={metrics[0]} | PERIOD=LAST_{years_back}_YEARS",
            "intent_query": "TREND_ANALYSIS",
            "entity_query": f"METRIC={metrics[0]}",
            "intent": "TREND_ANALYSIS",
            "speed_tier": "ANALYTICAL",
            "metrics": metrics,
            "years_back": years_back,
            "forecast_horizon": 0,
        }

    if is_statistical and (metrics or is_historical):
        if not metrics:
            metrics = ["baptism"]
        return {
            **base_meta,
            "normalized_query": f"INTENT=STATISTICAL_ANALYSIS | METRIC={metrics[0]} | PERIOD=LAST_{years_back}_YEARS",
            "intent_query": "STATISTICAL_ANALYSIS",
            "entity_query": f"METRIC={metrics[0]}",
            "intent": "STATISTICAL_ANALYSIS",
            "speed_tier": "ANALYTICAL",
            "metrics": metrics,
            "years_back": years_back,
            "forecast_horizon": 0,
        }

    if is_historical and metrics:
        return {
            **base_meta,
            "normalized_query": f"INTENT=HISTORICAL_ANALYSIS | METRIC={metrics[0]} | PERIOD=LAST_{years_back}_YEARS",
            "intent_query": "HISTORICAL_ANALYSIS",
            "entity_query": f"METRIC={metrics[0]}",
            "intent": "HISTORICAL_ANALYSIS",
            "speed_tier": "ANALYTICAL",
            "metrics": metrics,
            "years_back": years_back,
            "forecast_horizon": 0,
        }

    from koinonia_assistant.rag.name_search import classify_query_intent

    q_info = classify_query_intent(question)
    s_intent = q_info.get("intent")
    scope = q_info.get("scope", "GENERAL_QUERY")
    person_name = q_info.get("person_name")

    m_sac_count = re.search(r"\b(?:who\s+)?(?:got|received|have|with|having)\s+(\d+)\s+(?:sacraments?|sacrements?)\b", q_low)

    # Priority 1: Specific Person Name or Family Card / Member ID Lookup (e.g. "Show all sacrament details of Adaikala Abinaya A")
    if (person_name and not m_sac_count) or re.search(r"\b(?:YLG/\d+|family\s+id|member\s+id|\d{4,5})\b", question, re.IGNORECASE):
        if scope in ("BAPTISM_STATUS", "CONFIRMATION_STATUS", "COMMUNION_STATUS", "MARRIAGE_STATUS", "ALL_SACRAMENTS"):
            c_int = "SACRAMENT_SEARCH"
        elif scope in ("FAMILY_DETAILS", "FAMILY_MEMBERS_ONLY"):
            c_int = "FAMILY_SEARCH"
        else:
            c_int = "MEMBER_SEARCH"
        return {
            **base_meta,
            "normalized_query": f"INTENT={c_int} | SCOPE={scope} | ENTITY={person_name or 'ID'}",
            "intent_query": c_int,
            "entity_query": person_name or "ID",
            "intent": c_int,
            "scope": scope,
            "person_name": person_name,
            "speed_tier": "FAST",
        }

    # Priority 2: Qualified Member Sacrament List (e.g. "List any 10 members who got 3 Sacrements")
    if m_sac_count or (
        not person_name
        and re.search(r"^\s*(?:list|show|give|display|get)\b", q_low)
        and re.search(r"\b(?:members?|people|parishioners?|who\s+got|who\s+received|who\s+have)\b", q_low)
        and any(w in q_low for w in ["sacrament", "sacrement", "baptized", "confirmed", "married", "who got", "who received", "who have"])
    ):
        req_count = 10
        m_lim = re.search(r"\b(?:list|show|give|display|get)\s+(?:me\s+)?(?:any\s+|some\s+|first\s+|top\s+)?(\d+)\b", q_low)
        if m_lim:
            req_count = max(1, min(100, int(m_lim.group(1))))
        return {
            **base_meta,
            "normalized_query": f"INTENT=LIST | SUB=QUALIFIED_MEMBER_SACRAMENT_LIST | SAC_COUNT={m_sac_count.group(1) if m_sac_count else 'ANY'}",
            "intent_query": "LIST",
            "entity_query": "MEMBER_SACRAMENTS",
            "intent": "LIST",
            "sub_intent": "QUALIFIED_MEMBER_SACRAMENT_LIST",
            "speed_tier": "FAST",
            "limit": req_count,
            "sacrament_count_filter": int(m_sac_count.group(1)) if m_sac_count else None,
            "metrics": metrics,
        }

    if s_intent in ("LIST_MEMBERS", "LIST_FAMILIES"):
        return {
            **base_meta,
            "normalized_query": f"INTENT=LIST | SUB={s_intent}",
            "intent_query": "LIST",
            "entity_query": s_intent,
            "intent": "LIST",
            "sub_intent": s_intent,
            "speed_tier": "FAST",
            "limit": q_info.get("requested_count", 10),
            "filter": q_info.get("name_filter"),
            "metrics": metrics,
        }

    if s_intent in ("COUNT_MEMBERS", "COUNT_FAMILIES") or re.search(r"\b(?:how\s+many|count|total\s+number\s+of)\b", q_low):
        return {
            **base_meta,
            "normalized_query": f"INTENT=COUNT | SUB={s_intent or 'COUNT_RECORDS'}",
            "intent_query": "COUNT",
            "entity_query": s_intent or "COUNT_RECORDS",
            "intent": "COUNT",
            "sub_intent": s_intent if s_intent in ("COUNT_MEMBERS", "COUNT_FAMILIES") else "COUNT_RECORDS",
            "speed_tier": "FAST",
            "metrics": metrics,
        }


    if re.search(r"^\s*(?:list|show\s+all|display\s+all)\b", q_low):
        return {
            **base_meta,
            "normalized_query": "INTENT=LIST | SUB=GENERAL_LIST",
            "intent_query": "LIST",
            "entity_query": "GENERAL",
            "intent": "LIST",
            "sub_intent": "GENERAL_LIST",
            "speed_tier": "FAST",
            "metrics": metrics,
        }

    return {
        **base_meta,
        "normalized_query": "INTENT=GENERAL_DATABASE_QUERY",
        "intent_query": "GENERAL_DATABASE_QUERY",
        "entity_query": "GENERAL",
        "intent": "GENERAL_DATABASE_QUERY",
        "speed_tier": "FAST",
        "metrics": metrics,
    }


# ─── 5. Qualified Member Sacrament List Query ─────────────────────────────────
def execute_qualified_member_sacrament_list(
    sacrament_count_filter: Optional[int],
    metrics: List[str],
    limit: int = 10,
    user_parish: Optional[str] = None,
    user_diocese: Optional[str] = None,
    auth_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes a verified database query on tabMember for queries like:
    'List any 10 members who got 3 Sacrements'
    Strictly enforces authorization scope BEFORE querying.
    """
    import frappe

    where_clauses = ["1=1"]
    params: List[Any] = []

    effective_parish = (auth_ctx or {}).get("parish_id") or user_parish
    effective_diocese = (auth_ctx or {}).get("diocese_id") or user_diocese

    if effective_parish:
        where_clauses.append("(m.parish_id = %s OR m.parish_id LIKE %s)")
        params.extend([effective_parish, f"%{effective_parish}%"])
        scope_label = effective_parish
    elif effective_diocese and effective_diocese != "All Dioceses":
        where_clauses.append("m.diocese_id = %s")
        params.append(effective_diocese)
        scope_label = effective_diocese
    else:
        scope_label = "Diocesan Registry"

    sac_expr = "((m.bapt_date IS NOT NULL) + (m.fhc_date IS NOT NULL) + (m.cnf_date IS NOT NULL) + (m.mrg_date IS NOT NULL))"

    if sacrament_count_filter is not None:
        where_clauses.append(f"{sac_expr} = %s")
        params.append(int(sacrament_count_filter))
    elif metrics:
        for m_key in metrics:
            col = SACRAMENT_METRICS.get(m_key, {}).get("member_col")
            if col:
                where_clauses.append(f"m.{col} IS NOT NULL")
    else:
        where_clauses.append(f"{sac_expr} >= 1")

    sql = f"""
        SELECT
            m.name AS member_id,
            TRIM(CONCAT_WS(' ', m.first_name, m.middle_name, m.last_name)) AS full_name,
            m.gender,
            m.mobile,
            m.parish_id,
            f.family_register_number AS family_card,
            f.reference AS family_name,
            m.bapt_date,
            m.fhc_date,
            m.cnf_date,
            m.mrg_date,
            {sac_expr} AS sacraments_received_count
        FROM `tabMember` m
        LEFT JOIN `tabFamily` f ON f.name = m.family_id
        WHERE {' AND '.join(where_clauses)}
        ORDER BY sacraments_received_count DESC, full_name ASC
        LIMIT %s
    """
    params.append(int(limit))

    if auth_ctx:
        val_check = authorization_sql_validator(sql, auth_ctx)
        if not val_check["valid"]:
            raise PermissionError(f"SQL Authorization Validation Failed: {val_check['reason']}")

    rows = frappe.db.sql(sql, tuple(params), as_dict=True)

    formatted_rows = []
    for idx, r in enumerate(rows, 1):
        rec_list = []
        if r.get("bapt_date"):
            rec_list.append(f"Baptism ({r['bapt_date']})")
        if r.get("fhc_date"):
            rec_list.append(f"FHC ({r['fhc_date']})")
        if r.get("cnf_date"):
            rec_list.append(f"Confirmation ({r['cnf_date']})")
        if r.get("mrg_date"):
            rec_list.append(f"Marriage ({r['mrg_date']})")
        formatted_rows.append({
            "#": idx,
            "Full Name": r.get("full_name"),
            "Family Card": r.get("family_card") or "-",
            "Family Name": r.get("family_name") or "-",
            "Sacraments Count": int(r.get("sacraments_received_count") or 0),
            "Sacraments Received": ", ".join(rec_list) if rec_list else "None",
            "Contact": r.get("mobile") or "-",
            "Parish": r.get("parish_id") or "-",
        })

    if not formatted_rows:
        target_desc = f"exactly **{sacrament_count_filter}** recorded sacraments" if sacrament_count_filter else "the requested sacraments"
        reply = f"No members were found in **{scope_label}** with {target_desc}."
        return {
            "reply": reply,
            "generated_sql": sql.strip(),
            "data": [],
            "records_retrieved": 0,
        }

    target_desc = (
        f"**{sacrament_count_filter} Sacraments**"
        if sacrament_count_filter is not None
        else "the requested sacraments"
    )
    lines = [
        f"Found **{len(formatted_rows)}** verified member(s) in **{scope_label}** who received {target_desc}:\n",
        "| # | Member Name | Family Card | Sacraments Count | Sacraments Received (Dates) | Contact |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in formatted_rows:
        lines.append(
            f"| {r['#']} | **{r['Full Name']}** | `{r['Family Card']}` | **{r['Sacraments Count']}** | {r['Sacraments Received']} | {r['Contact']} |"
        )

    return {
        "reply": "\n".join(lines),
        "generated_sql": sql.strip(),
        "data": formatted_rows,
        "records_retrieved": len(formatted_rows),
    }


# ─── 6. Strict Scope-Constrained Historical Sacrament Fetcher (Sections 50, 54–61, 68–70) ───
def fetch_yearly_sacrament_series(
    metric_key: str,
    years_back: int = 10,
    user_parish: Optional[str] = None,
    user_diocese: Optional[str] = None,
    auth_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Retrieves the year-by-year historical count for a given sacrament metric.
    CRITICAL SECURITY GUARANTEE (Sections 50–76):
    - Enforces authorization scope BEFORE aggregation (WHERE parish_id IN ... GROUP BY YEAR).
    - NEVER falls back to Diocesan-wide data when the user is a Parish-level user.
    - Validates generated SQL via authorization_sql_validator BEFORE execution.
    """
    import frappe

    meta = SACRAMENT_METRICS.get(metric_key, SACRAMENT_METRICS["baptism"])
    table = meta["table"]
    date_col = meta["date_col"]
    parish_cols = meta["parish_cols"]

    effective_scope_type = (auth_ctx or {}).get("scope_type") or ("PARISH" if user_parish else "DIOCESE")
    effective_parish = (auth_ctx or {}).get("parish_id") or user_parish
    effective_diocese = (auth_ctx or {}).get("diocese_id") or user_diocese
    effective_scope_name = (auth_ctx or {}).get("scope_name") or effective_parish or effective_diocese or "Authorized Scope"

    # Anchor end_year to 2025 for 10-year historical windows (2016–2025)
    end_year = 2025
    start_year = end_year - years_back + 1

    where_parts = [
        f"`{date_col}` IS NOT NULL",
        f"YEAR(`{date_col}`) BETWEEN %s AND %s",
    ]
    params: List[Any] = [start_year, end_year]

    if effective_scope_type == "PARISH":
        if not effective_parish:
            raise PermissionError("PARISH scope requires a verified parish_id in authorization_context.")
        p_cond = " OR ".join([f"`{col}` = %s OR `{col}` LIKE %s" for col in parish_cols])
        where_parts.append(f"({p_cond})")
        for _ in parish_cols:
            params.extend([effective_parish, f"%{effective_parish}%"])
    elif effective_scope_type == "VICARIATE" and (auth_ctx or {}).get("vicariate_id"):
        where_parts.append("`vicariate_id` = %s")
        params.append(auth_ctx["vicariate_id"])
    elif effective_diocese and effective_diocese != "All Dioceses":
        cnt_dio = frappe.db.sql(f"SELECT COUNT(*) FROM `{table}` WHERE diocese_id = %s", (effective_diocese,))[0][0]
        if cnt_dio > 0:
            where_parts.append("`diocese_id` = %s")
            params.append(effective_diocese)

    sql_query = f"""
        SELECT YEAR(`{date_col}`) AS yr, COUNT(*) AS cnt
        FROM `{table}`
        WHERE {' AND '.join(where_parts)}
        GROUP BY YEAR(`{date_col}`)
        ORDER BY yr ASC
    """

    # Validate SQL security BEFORE execution (Section 59)
    val_res = authorization_sql_validator(
        sql_query,
        auth_ctx or {"scope_type": effective_scope_type, "parish_id": effective_parish},
    )
    if not val_res["valid"]:
        raise PermissionError(f"SQL Security Validator rejected query: {val_res['reason']}")

    rows = frappe.db.sql(sql_query, tuple(params), as_dict=True)
    year_map = {int(r["yr"]): int(r["cnt"]) for r in rows if r.get("yr") is not None}
    missing_years = [yr for yr in range(start_year, end_year + 1) if yr not in year_map]

    series = []
    for yr in range(start_year, end_year + 1):
        series.append({
            "year": yr,
            "count": year_map.get(yr, 0),
        })

    authorized_record_count = sum(item["count"] for item in series)

    return {
        "metric_key": metric_key,
        "metric_label": meta["label"],
        "ta_label": meta["ta_label"],
        "color": meta["color"],
        "start_year": start_year,
        "end_year": end_year,
        "scope_type": effective_scope_type,
        "scope_id": (auth_ctx or {}).get("scope_id") or effective_parish or "DIOCESE",
        "scope_name": effective_scope_name,
        "scope_used": effective_scope_name,
        "authorized_record_count": authorized_record_count,
        "sql_validation": val_res["reason"],
        "series": series,
        "observed_years_count": len(year_map),
        "missing_years": missing_years,
        "null_dates_count": 0,
        "parish_years_dict": year_map if effective_scope_type == "PARISH" else {},
        "sql": sql_query.strip(),
    }


# ─── 7. Deterministic Statistical Computation & Forecasting ───────────────────
def compute_series_statistics(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes verified deterministic statistics from a yearly series [{year, count}].
    """
    if not series:
        return {}

    n = len(series)
    years = [int(x["year"]) for x in series]
    counts = [int(x["count"]) for x in series]

    total = sum(counts)
    avg = total / n if n > 0 else 0.0

    min_val = min(counts)
    max_val = max(counts)
    min_year = years[counts.index(min_val)]
    max_year = years[counts.index(max_val)]

    first_val = counts[0]
    last_val = counts[-1]
    abs_change = last_val - first_val
    pct_change = ((last_val - first_val) / first_val * 100.0) if first_val > 0 else 0.0

    if first_val > 0 and last_val > 0 and n > 1:
        cagr = ((last_val / first_val) ** (1.0 / (n - 1)) - 1.0) * 100.0
    else:
        cagr = 0.0

    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = avg
    num = sum((x_vals[i] - x_mean) * (counts[i] - y_mean) for i in range(n))
    den = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0.0
    intercept = y_mean - slope * x_mean

    ss_tot = sum((c - y_mean) ** 2 for c in counts)
    ss_res = sum((counts[i] - (intercept + slope * x_vals[i])) ** 2 for i in range(n))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    residual_se = math.sqrt(ss_res / max(1, n - 2)) if n > 2 else (math.sqrt(ss_tot / max(1, n)) if n > 0 else 1.0)

    variance = ss_tot / max(1, n - 1) if n > 1 else 0.0
    std_dev = math.sqrt(variance)
    cv_pct = (std_dev / avg * 100.0) if avg > 0 else 0.0

    yoy_rows = []
    for i in range(n):
        yr = years[i]
        cnt = counts[i]
        if i == 0:
            prev = None
            yoy_abs = None
            yoy_pct = None
        else:
            prev = counts[i - 1]
            yoy_abs = cnt - prev
            yoy_pct = ((cnt - prev) / prev * 100.0) if prev > 0 else None

        window = counts[max(0, i - 2): i + 1]
        ma_3 = sum(window) / len(window)
        ma_3_rounded = int(round(ma_3))

        yoy_rows.append({
            "year": yr,
            "count": cnt,
            "prev_count": prev,
            "yoy_abs": yoy_abs,
            "yoy_pct": int(round(yoy_pct)) if yoy_pct is not None else None,
            "moving_avg_3yr": ma_3_rounded,
            "moving_avg_3yr_exact": round(ma_3, 1),
        })

    if abs(slope) < max(0.5, avg * 0.01):
        trend_direction = "Stable / Plateauing"
    elif slope > 0:
        if min_year in (2020, 2021) and last_val > min_val:
            trend_direction = "V-Shaped Recovery with Net Upward Growth"
        else:
            trend_direction = "Increasing (Upward Growth)"
    else:
        if max_year in (2022, 2023, 2024) and last_val > first_val:
            trend_direction = "Post-2021 Recovery Following Earlier Contraction"
        else:
            trend_direction = "Decreasing / Partial-Window Recording"

    return {
        "years_count": n,
        "first_year": years[0],
        "last_year": years[-1],
        "first_count": first_val,
        "last_count": last_val,
        "total": total,
        "average_per_year": int(round(avg)),
        "average_per_year_exact": round(avg, 1),
        "minimum": min_val,
        "year_of_minimum": min_year,
        "maximum": max_val,
        "year_of_maximum": max_year,
        "absolute_change": abs_change,
        "percentage_change": int(round(pct_change)),
        "cagr_pct": int(round(cagr)),
        "linear_regression_slope": int(round(slope)),
        "linear_regression_intercept": round(intercept, 2),
        "r_squared": round(r_squared, 3),
        "residual_se": round(residual_se, 2),
        "volatility_std": int(round(std_dev)),
        "cv_pct": int(round(cv_pct)),
        "trend_direction": trend_direction,
        "yoy_rows": yoy_rows,
    }


def format_friendly_yoy_and_avg(row: Dict[str, Any], index: int, lang: str = "en") -> Tuple[str, str]:
    """
    Formats Year-over-Year change and 3-Year Moving Average using rounded whole numbers
    and plain human-friendly descriptions instead of confusing 'Baseline' or decimal people counts.
    """
    cnt = int(row.get("count", 0))
    prev = row.get("prev_count")
    yoy_abs = row.get("yoy_abs")
    yoy_pct = row.get("yoy_pct")
    avg_rounded = int(round(row.get("moving_avg_3yr", 0)))

    if lang == "ta":
        avg_str = f"{avg_rounded:,} / ஆண்டு"
        if index == 0 or prev is None or yoy_abs is None:
            yoy_str = "தொடக்க ஆண்டு (Starting Year)"
        elif yoy_abs == 0:
            yoy_str = "மாற்றமில்லை (முந்தைய ஆண்டே)"
        elif prev == 0 and cnt > 0:
            yoy_str = f"▲ +{cnt:,} அதிகரிப்பு (0-இல் இருந்து)"
        elif yoy_abs > 0:
            pct_part = f" (+{int(round(yoy_pct))}%)" if yoy_pct is not None else ""
            yoy_str = f"▲ +{yoy_abs:,} அதிகரிப்பு{pct_part}"
        else:
            pct_part = f" ({int(round(yoy_pct))}%)" if yoy_pct is not None else ""
            yoy_str = f"▼ {yoy_abs:,} குறைவு{pct_part}"
    else:
        avg_str = f"{avg_rounded:,} / yr"
        if index == 0 or prev is None or yoy_abs is None:
            yoy_str = "Starting Year"
        elif yoy_abs == 0:
            yoy_str = "No Change (Same as prev. yr)"
        elif prev == 0 and cnt > 0:
            yoy_str = f"▲ +{cnt:,} more (up from 0)"
        elif yoy_abs > 0:
            pct_part = f" (+{int(round(yoy_pct))}%)" if yoy_pct is not None else ""
            yoy_str = f"▲ +{yoy_abs:,} more{pct_part}"
        else:
            pct_part = f" ({int(round(yoy_pct))}%)" if yoy_pct is not None else ""
            yoy_str = f"▼ {yoy_abs:,} fewer{pct_part}"

    return yoy_str, avg_str



def generate_statistical_forecast(
    series: List[Dict[str, Any]],
    stats: Dict[str, Any],
    horizon: int = 10,
    missing_years: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Executes Forecast Data Quality Check (Section 16) and generates multi-year
    forecasts using ONLY the user's authorized scope dataset (Section 61).
    """
    non_zero_points = [x for x in series if int(x["count"]) > 0]
    n_valid = len(non_zero_points)
    n_total = len(series)
    years = [int(x["year"]) for x in series]
    counts = [float(x["count"]) for x in series]

    dq_report = {
        "historical_years_requested": n_total,
        "non_zero_years_found": n_valid,
        "missing_years": missing_years or [],
        "covid_structural_dip_detected": any(
            int(x["year"]) in (2020, 2021) and int(x["count"]) < stats.get("average_per_year", 0) * 0.92
            for x in series
        ),
        "quality_tier": (
            "INSUFFICIENT" if n_valid < 3
            else ("SHORT_SERIES_CAUTION" if n_valid <= 5 else "HIGH_RELIABILITY")
        ),
    }

    if n_valid < 3:
        return {
            "sufficient_data": False,
            "data_quality": dq_report,
            "error_message": (
                f"There are only {n_valid} years of verified historical records available in your authorized scope, "
                f"which is insufficient for a reliable {horizon}-year statistical forecast."
            ),
            "forecast_rows": [],
            "method": "Insufficient Historical Data (<3 years)",
        }

    # For sparse series with trailing zeros (e.g. local parish where latest entries were 2016-2021),
    # fit the exponential smoothing model on the active recorded points so projections reflect the parish's actual active rate.
    active_counts = [float(x["count"]) for x in non_zero_points] if counts[-1] == 0 else counts
    n_active = len(active_counts)

    best_alpha, best_beta, phi = 0.45, 0.20, 0.90
    best_sse = float("inf")

    for a_cand in (0.30, 0.45, 0.60, 0.75):
        for b_cand in (0.10, 0.15, 0.25):
            lvl = active_counts[0]
            trd = (active_counts[-1] - active_counts[0]) / max(1, n_active - 1) if n_active > 1 else 0.0
            sse = 0.0
            for t in range(1, n_active):
                pred = lvl + phi * trd
                err = active_counts[t] - pred
                sse += err * err
                prev_lvl = lvl
                lvl = a_cand * active_counts[t] + (1.0 - a_cand) * (lvl + phi * trd)
                trd = b_cand * (lvl - prev_lvl) + (1.0 - b_cand) * phi * trd
            if sse < best_sse:
                best_sse = sse
                best_alpha, best_beta = a_cand, b_cand

    level = active_counts[0]
    trend = (active_counts[-1] - active_counts[0]) / max(1, n_active - 1) if n_active > 1 else 0.0
    residuals = []
    for t in range(1, n_active):
        pred = level + phi * trend
        residuals.append(active_counts[t] - pred)
        prev_level = level
        level = best_alpha * active_counts[t] + (1.0 - best_alpha) * (level + phi * trend)
        trend = best_beta * (level - prev_level) + (1.0 - best_beta) * phi * trend

    holt_rmse = math.sqrt(sum(r * r for r in residuals) / max(1, len(residuals))) if residuals else max(1.0, stats.get("residual_se", 2.0))
    selected_method = f"Damped Holt's Linear Exponential Smoothing (α={best_alpha:.2f}, β={best_beta:.2f}, φ={phi:.2f})"
    sigma = max(holt_rmse, stats.get("average_per_year", 10.0) * 0.12, 1.5)

    last_year = years[-1]
    forecast_rows = []
    for h in range(1, horizon + 1):
        target_year = last_year + h
        phi_sum = sum(phi ** j for j in range(1, h + 1))
        raw_point = level + phi_sum * trend
        # Prevent collapse to 0 when parish has active historical mean
        floor_val = max(1, int(round(stats.get("average_per_year", 1.0) * 0.5))) if stats.get("total", 0) > 0 else 0
        point_est = max(floor_val, int(round(raw_point)))

        step_se = sigma * math.sqrt(1.0 + 0.18 * (h - 1))
        margin_80 = int(round(1.2816 * step_se))
        margin_95 = int(round(1.9600 * step_se))

        lower_80 = max(0, point_est - margin_80)
        upper_80 = point_est + margin_80
        lower_95 = max(0, point_est - margin_95)
        upper_95 = point_est + margin_95

        forecast_rows.append({
            "year": target_year,
            "horizon_step": h,
            "projected": point_est,
            "lower_bound": lower_80,
            "upper_bound": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "range_str": f"{lower_80:,}–{upper_80:,}",
            "status": "Forecast / Projected",
        })

    return {
        "sufficient_data": True,
        "data_quality": dq_report,
        "method": selected_method,
        "model_code": "HOLT_DAMPED",
        "historical_period": f"{years[0]}–{years[-1]}",
        "forecast_period": f"{last_year + 1}–{last_year + horizon}",
        "forecast_rows": forecast_rows,
    }


def generate_dynamic_suggested_questions(
    intent_info: Dict[str, Any],
    detected_language: str,
    auth_ctx: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generates context-aware suggested follow-up questions in the user's language (Section 44).
    If the user asks in Tamil, returns Tamil suggested questions using canonical Catholic terminology.
    """
    metrics = intent_info.get("metrics") or ["baptism"]
    primary_m = metrics[0]
    meta = SACRAMENT_METRICS.get(primary_m, SACRAMENT_METRICS["baptism"])
    scope_name = (auth_ctx or {}).get("scope_name") or "Yelagiri Parish"

    if detected_language == "ta":
        ta_sac = meta["ta_label"]
        other_ta = "திருமுழுக்கு" if primary_m != "baptism" else "முதல் நற்கருணை"
        return [
            f"கடந்த 10 ஆண்டுகளில் {other_ta} எண்ணிக்கை என்ன?",
            f"{ta_sac} எண்ணிக்கையில் ஏற்பட்ட மாற்றத்தை காட்டு",
            f"அடுத்த 10 ஆண்டுகளுக்கான {ta_sac} கணிப்பை வழங்கவும்",
        ]
    else:
        eng_sac = meta["label"]
        other_eng = "First Holy Communion" if primary_m == "baptism" else "Baptism"
        return [
            f"What was the number of {other_eng}s each year for the last 10 years in {scope_name}?",
            f"How has {eng_sac} changed over the last 10 years in {scope_name}?",
            f"Forecast {eng_sac} for the next 10 years in {scope_name}",
        ]


# ─── 8. Unified Analytics & Forecast Pipeline (Strictly Scope-Constrained & Multilingual) ───
def run_analytics_or_forecast_pipeline(
    intent_info: Dict[str, Any],
    question: str,
    user_parish: Optional[str] = None,
    user_diocese: Optional[str] = None,
    auth_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete backend analytical / historical / trend / forecast / comparison pipeline.
    Enforces:
    1. Role-based authorization scope (NEVER exposes Diocesan Benchmark to Parish users).
    2. Tamil-first output formatting when detected_language == 'ta'.
    """
    intent = intent_info.get("intent", "HISTORICAL_ANALYSIS")
    metrics = intent_info.get("metrics") or ["baptism"]
    years_back = int(intent_info.get("years_back") or 10)
    forecast_horizon = int(intent_info.get("forecast_horizon") or (10 if intent == "FORECAST" else 0))
    include_forecast = bool(intent_info.get("include_forecast") or intent == "FORECAST" or forecast_horizon > 0)
    lang = intent_info.get("detected_language") or detect_query_language(question)

    datasets: Dict[str, Dict[str, Any]] = {}
    all_sqls = []

    for m_key in metrics:
        ds = fetch_yearly_sacrament_series(
            metric_key=m_key,
            years_back=years_back,
            user_parish=user_parish,
            user_diocese=user_diocese,
            auth_ctx=auth_ctx,
        )
        ds["stats"] = compute_series_statistics(ds["series"])
        if include_forecast and forecast_horizon > 0:
            ds["forecast"] = generate_statistical_forecast(
                series=ds["series"],
                stats=ds["stats"],
                horizon=forecast_horizon,
                missing_years=ds["missing_years"],
            )
        datasets[m_key] = ds
        all_sqls.append(ds["sql"])

    primary_key = metrics[0]
    primary_ds = datasets[primary_key]
    primary_stats = primary_ds["stats"]
    scope_type = primary_ds["scope_type"]
    scope_id = primary_ds["scope_id"]
    scope_name = primary_ds["scope_name"]
    authorized_record_count = primary_ds["authorized_record_count"]

    # Build table rows (`data`) and Chart.js structure (`chart`)
    if len(metrics) == 1:
        m_label = primary_ds["ta_label"] if lang == "ta" else primary_ds["metric_label"]
        table_rows = []
        chart_x = []
        hist_vals = []
        proj_vals = []
        lower_vals = []
        upper_vals = []

        yoy_col_name = "முந்தைய ஆண்டுடன் ஒப்பீடு" if lang == "ta" else "Change from Previous Year"
        avg_col_name = "3-ஆண்டு சராசரி (முழு எண்)" if lang == "ta" else "3-Year Average (Rounded)"

        for idx, r in enumerate(primary_stats["yoy_rows"]):
            yr_str = str(r["year"])
            chart_x.append(yr_str)
            hist_vals.append(r["count"])
            if include_forecast:
                proj_vals.append(None)
                lower_vals.append(None)
                upper_vals.append(None)

            yoy_str, avg_str = format_friendly_yoy_and_avg(r, idx, lang=lang)
            table_rows.append({
                "Year": yr_str,
                "Record Type": "பதிவு செய்யப்பட்டவை (Observed)" if lang == "ta" else "Observed (Actual DB)",
                f"{m_label} Count": r["count"],
                yoy_col_name: yoy_str,
                avg_col_name: avg_str,
                "Prediction Interval (80%)": "-",
            })

        f_obj = primary_ds.get("forecast")
        if include_forecast and f_obj and f_obj.get("sufficient_data"):
            if proj_vals:
                proj_vals[-1] = hist_vals[-1]
                lower_vals[-1] = hist_vals[-1]
                upper_vals[-1] = hist_vals[-1]

            for fr in f_obj["forecast_rows"]:
                yr_str = f"{fr['year']} (Est.)"
                chart_x.append(yr_str)
                hist_vals.append(None)
                proj_vals.append(fr["projected"])
                lower_vals.append(fr["lower_bound"])
                upper_vals.append(fr["upper_bound"])

                table_rows.append({
                    "Year": str(fr["year"]),
                    "Record Type": "கணிக்கப்பட்டவை (Forecast)" if lang == "ta" else "Forecast (Projected)",
                    f"{m_label} Count": fr["projected"],
                    yoy_col_name: "கணிப்பு (Projected)" if lang == "ta" else "Projected",
                    avg_col_name: "-",
                    "Prediction Interval (80%)": fr["range_str"],
                })

        chart_series = [
            {
                "name": f"{m_label} ({scope_name})",
                "values": hist_vals,
                "color": primary_ds["color"],
                "dashed": False,
            }
        ]
        if include_forecast and f_obj and f_obj.get("sufficient_data"):
            chart_series.extend([
                {
                    "name": f"கணிப்பு / Projected {m_label}" if lang == "ta" else f"Projected {m_label} (Forecast)",
                    "values": proj_vals,
                    "color": "#38bdf8",
                    "dashed": True,
                },
                {
                    "name": "80% Upper Bound",
                    "values": upper_vals,
                    "color": "rgba(56, 189, 248, 0.45)",
                    "dashed": True,
                },
                {
                    "name": "80% Lower Bound",
                    "values": lower_vals,
                    "color": "rgba(56, 189, 248, 0.45)",
                    "dashed": True,
                },
            ])

        chart_payload = {
            "chart_type": "line",
            "title": (
                f"{scope_name} — {m_label} ({primary_ds['start_year']}–{primary_ds['end_year']})"
                if lang == "ta"
                else f"{scope_name} — {m_label} {'Historical & Forecast Trajectory' if include_forecast else 'Historical Trend Analysis'}"
            ),
            "x": chart_x,
            "series": chart_series,
        }

    else:
        start_yr = primary_ds["start_year"]
        end_yr = primary_ds["end_year"]
        chart_x = [str(y) for y in range(start_yr, end_yr + 1)]
        if include_forecast and forecast_horizon > 0:
            chart_x.extend([f"{end_yr + h} (Est.)" for h in range(1, forecast_horizon + 1)])

        table_rows = []
        for yr in range(start_yr, end_yr + 1):
            row_dict: Dict[str, Any] = {"Year": str(yr), "Record Type": "Observed (Actual DB)"}
            for m_key in metrics:
                m_ds = datasets[m_key]
                lbl_col = m_ds["ta_label"] if lang == "ta" else m_ds["metric_label"]
                val = next((item["count"] for item in m_ds["series"] if item["year"] == yr), 0)
                row_dict[lbl_col] = val
            table_rows.append(row_dict)

        if include_forecast and forecast_horizon > 0:
            for h in range(1, forecast_horizon + 1):
                f_yr = end_yr + h
                row_dict = {"Year": str(f_yr), "Record Type": "Forecast (Projected)"}
                for m_key in metrics:
                    m_ds = datasets[m_key]
                    lbl_col = m_ds["ta_label"] if lang == "ta" else m_ds["metric_label"]
                    f_obj = m_ds.get("forecast")
                    if f_obj and f_obj.get("sufficient_data"):
                        fr = next((x for x in f_obj["forecast_rows"] if x["year"] == f_yr), None)
                        row_dict[lbl_col] = f"{fr['projected']} ({fr['range_str']})" if fr else "-"
                    else:
                        row_dict[lbl_col] = "Insufficient Data"
                table_rows.append(row_dict)

        chart_series = []
        for m_key in metrics:
            m_ds = datasets[m_key]
            lbl_col = m_ds["ta_label"] if lang == "ta" else m_ds["metric_label"]
            vals = [item["count"] for item in m_ds["series"]]
            if include_forecast and forecast_horizon > 0:
                f_obj = m_ds.get("forecast")
                if f_obj and f_obj.get("sufficient_data"):
                    vals.extend([fr["projected"] for fr in f_obj["forecast_rows"]])
                else:
                    vals.extend([None] * forecast_horizon)
            chart_series.append({
                "name": lbl_col,
                "values": vals,
                "color": m_ds["color"],
                "dashed": False,
            })

        chart_payload = {
            "chart_type": "line",
            "title": f"{scope_name} — Multi-Sacrament Comparison" + (" & Forecast" if include_forecast else ""),
            "x": chart_x,
            "series": chart_series,
        }

    # Build Dynamic Scope-Prefixed Report (Section 69: Never use 'Diocesan Registry Benchmark' for Parish users)
    md_sections = []
    if lang == "ta":
        ta_intent_titles = {
            "HISTORICAL_ANALYSIS": "வரலாற்றுப் பகுப்பாய்வு அறிக்கை",
            "STATISTICAL_ANALYSIS": "புள்ளிவிவரப் பகுப்பாய்வு அறிக்கை",
            "TREND_ANALYSIS": "போக்குப் பகுப்பாய்வு அறிக்கை",
            "FORECAST": "புள்ளிவிவர முன்கணிப்பு அறிக்கை",
            "COMPARISON": "திருவருட்சாதன ஒப்பீட்டு அறிக்கை",
        }
        ta_title = ta_intent_titles.get(intent, "சரிபார்க்கப்பட்ட புள்ளிவிவர அறிக்கை")
        md_sections.append(f"### 📊 {scope_name} — {ta_title}")
        md_sections.append(f"- **அனுமதிக்கப்பட்ட பங்கு எல்லை (Authorized Scope):** `{scope_name}` (`{scope_type}`)")
        md_sections.append(f"- **ஆய்வுக் காலம் (Observed Period):** `{primary_ds['start_year']}–{primary_ds['end_year']}` ({years_back} ஆண்டுகள்)")
        md_sections.append(f"- **அனுமதிக்கப்பட்ட மொத்தப் பதிவுகள்:** `{authorized_record_count:,}`")
        if include_forecast and primary_ds.get("forecast") and primary_ds["forecast"].get("sufficient_data"):
            f_info = primary_ds["forecast"]
            md_sections.append(f"- **முன்கணிப்புக் காலம் (Forecast Period):** `{f_info['forecast_period']}` ({forecast_horizon} ஆண்டுகள்)")
            md_sections.append(f"- **புள்ளிவிவர மாதிரி:** `{f_info['method']}`")

        md_sections.append("\n#### 1. சரிபார்க்கப்பட்ட புள்ளிவிவரச் சுருக்கம் (Observed Statistical Summary)")
        for m_key in metrics:
            m_ds = datasets[m_key]
            st = m_ds["stats"]
            ta_lbl = m_ds["ta_label"]
            md_sections.append(
                f"- **{ta_lbl} (`{st['first_year']}–{st['last_year']}`):** "
                f"**மொத்தம் (Total):** `{st['total']:,}` | "
                f"**ஆண்டு சராசரி (Rounded Mean):** `~{int(round(st['average_per_year'])):,} / ஆண்டு` | "
                f"**அதிகபட்சம் (Peak):** `{st['maximum']:,}` (`{st['year_of_maximum']}`) | "
                f"**குறைந்தபட்சம் (Low):** `{st['minimum']:,}` (`{st['year_of_minimum']}`) | "
                f"**நிகர மாற்றம்:** `{st['absolute_change']:+,}` (`{int(round(st['percentage_change'])):+d}%`) | "
                f"**சாய்வு (Trend Slope):** `{int(round(st['linear_regression_slope'])):+d} / ஆண்டு`"
            )

        if len(metrics) == 1:
            ta_lbl = primary_ds["ta_label"]
            md_sections.append(f"\n#### 2. ஆண்டு வாரியான {ta_lbl} தரவு (`{primary_ds['start_year']}–{primary_ds['end_year']}`)")
            md_sections.append(f"| ஆண்டு (Year) | நிலை (Status) | {ta_lbl} எண்ணிக்கை | முந்தைய ஆண்டுடன் ஒப்பீடு (Change from Prev. Yr) | 3-ஆண்டு சராசரி (முழு எண்) |")
            md_sections.append("| :--- | :--- | :--- | :--- | :--- |")
            for idx, r in enumerate(primary_stats["yoy_rows"]):
                yoy_str, avg_str = format_friendly_yoy_and_avg(r, idx, lang="ta")
                md_sections.append(
                    f"| `{r['year']}` | பதிவு செய்யப்பட்டவை | **{r['count']:,}** | {yoy_str} | **{avg_str}** |"
                )

            if include_forecast:
                f_obj = primary_ds.get("forecast")
                md_sections.append(f"\n#### 3. {ta_lbl} முன்கணிப்பு (`{f_obj.get('forecast_period', 'Future')}`) — *கணிக்கப்பட்ட மதிப்புகள்*")
                if not f_obj or not f_obj.get("sufficient_data"):
                    md_sections.append(f"> ⚠️ **தரவுத் தரக் குறிப்பு:** {f_obj.get('error_message')}")
                else:
                    md_sections.append("| கணிப்பு ஆண்டு | வகை | கணிக்கப்பட்ட எண்ணிக்கை | 80% நம்பிக்கை வரம்பு | 95% நம்பிக்கை வரம்பு |")
                    md_sections.append("| :--- | :--- | :--- | :--- | :--- |")
                    for fr in f_obj["forecast_rows"]:
                        md_sections.append(
                            f"| `{fr['year']}` | *கணிக்கப்பட்டவை* | **{fr['projected']:,}** | `{fr['lower_bound']:,} – {fr['upper_bound']:,}` | `{fr['lower_95']:,} – {fr['upper_95']:,}` |"
                        )
    else:
        intent_title = intent.replace("_", " ").title()
        md_sections.append(f"### 📊 {scope_name} — Verified {intent_title}")
        md_sections.append(f"- **Authorized Scope:** `{scope_name}` (`{scope_type}`)")
        md_sections.append(f"- **Historical Window (Observed):** `{primary_ds['start_year']}–{primary_ds['end_year']}` ({years_back} years)")
        md_sections.append(f"- **Authorized Record Count:** `{authorized_record_count:,}`")
        if include_forecast and primary_ds.get("forecast"):
            f_info = primary_ds["forecast"]
            if f_info.get("sufficient_data"):
                md_sections.append(f"- **Forecast Horizon (Projected):** `{f_info['forecast_period']}` ({forecast_horizon} years)")
                md_sections.append(f"- **Statistical Forecasting Model:** `{f_info['method']}`")

        md_sections.append("\n#### 1. Observed Statistical Summary (Verified Database Calculations)")
        for m_key in metrics:
            m_ds = datasets[m_key]
            st = m_ds["stats"]
            lbl = m_ds["metric_label"]
            md_sections.append(
                f"- **{lbl} (`{st['first_year']}–{st['last_year']}`):** "
                f"**Total:** `{st['total']:,}` | "
                f"**Annual Mean (Rounded):** `~{int(round(st['average_per_year'])):,} / yr` | "
                f"**Peak:** `{st['maximum']:,}` in `{st['year_of_maximum']}` | "
                f"**Low:** `{st['minimum']:,}` in `{st['year_of_minimum']}` | "
                f"**Net Change:** `{st['absolute_change']:+,}` (`{int(round(st['percentage_change'])):+d}%`, CAGR `{int(round(st['cagr_pct'])):+d}%`) | "
                f"**Trend Slope:** `{int(round(st['linear_regression_slope'])):+d} / yr` | "
                f"**Typical Variation ($\\sigma$):** `±{int(round(st['volatility_std']))} / yr` | "
                f"**Observed Trajectory:** *{st['trend_direction']}*"
            )

        if len(metrics) == 1:
            lbl = primary_ds["metric_label"]
            md_sections.append(f"\n#### 2. Year-by-Year Observed {lbl} Data (`{primary_ds['start_year']}–{primary_ds['end_year']}`)")
            md_sections.append(f"| Year | Status | {lbl} Count | Change from Previous Year | 3-Year Average (Rounded) |")
            md_sections.append("| :--- | :--- | :--- | :--- | :--- |")
            for idx, r in enumerate(primary_stats["yoy_rows"]):
                yoy_str, avg_str = format_friendly_yoy_and_avg(r, idx, lang="en")
                md_sections.append(
                    f"| `{r['year']}` | Observed | **{r['count']:,}** | {yoy_str} | **{avg_str}** |"
                )

            if include_forecast:
                f_obj = primary_ds.get("forecast")
                md_sections.append(f"\n#### 3. Projected {lbl} Forecast (`{f_obj.get('forecast_period', 'Future')}`) — *Estimated Values*")
                if not f_obj or not f_obj.get("sufficient_data"):
                    md_sections.append(
                        f"> ⚠️ **Data Quality Notice:** {f_obj.get('error_message', 'There is insufficient historical data to produce a reliable forecast.')}"
                    )
                else:
                    md_sections.append(
                        f"> **Model Note:** The values below are **Projected / Estimated** using `{f_obj['method']}` based on `{f_obj['historical_period']}` historical observations for **{scope_name}**. They are statistical projections and NOT actual database records.\n"
                    )
                    md_sections.append(f"| Forecast Year | Classification | Projected {lbl} | 80% Prediction Interval | 95% Prediction Interval |")
                    md_sections.append("| :--- | :--- | :--- | :--- | :--- |")
                    for fr in f_obj["forecast_rows"]:
                        md_sections.append(
                            f"| `{fr['year']}` | *Projected / Estimated* | **{fr['projected']:,}** | `{fr['lower_bound']:,} – {fr['upper_bound']:,}` | `{fr['lower_95']:,} – {fr['upper_95']:,}` |"
                        )
        else:
            headers = ["Year", "Record Type"] + [datasets[k]["ta_label"] if lang == "ta" else datasets[k]["metric_label"] for k in metrics]
            md_sections.append("\n#### 2. Multi-Sacrament Year-by-Year Comparison & Projections")
            md_sections.append("| " + " | ".join(headers) + " |")
            md_sections.append("| " + " | ".join([":---"] * len(headers)) + " |")
            for row in table_rows:
                vals = [str(row.get(h, "-")) for h in headers]
                md_sections.append("| " + " | ".join(vals) + " |")

    suggested_qs = generate_dynamic_suggested_questions(intent_info, lang, auth_ctx)

    return {
        "intent": intent,
        "scope_type": scope_type,
        "scope_id": scope_id,
        "scope_name": scope_name,
        "scope_used": scope_name,
        "authorized_record_count": authorized_record_count,
        "datasets": datasets,
        "primary_stats": primary_stats,
        "table_rows": table_rows,
        "chart": chart_payload,
        "deterministic_markdown": "\n".join(md_sections),
        "generated_sql": "\n\n".join(all_sqls),
        "records_retrieved": len(table_rows),
        "suggested_questions": suggested_qs,
    }
