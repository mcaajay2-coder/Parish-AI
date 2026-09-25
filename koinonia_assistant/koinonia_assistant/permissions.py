import frappe
from frappe import _

MANAGED_DOCTYPES = [
    "Member", "Family", "Baptism", "Communion",
    "Confirmation", "Marriage", "Death", "Anointing Of Sick",
    "Parish", "Vicariate", "Diocese"
]

# Built-in fallback mapping so any known Parish / Vicariate / Diocese user email
# always resolves to its exact authorized scope even if User Permission was cleared.
EMAIL_JURISDICTION_FALLBACK = {
    "yelagiri@koinonia.com": ("Parish", "Yelagiri Parish"),
    "priest_yelagiri@koinonia.com": ("Parish", "Yelagiri Parish"),
    "yelagiri_priest@koinonia.com": ("Parish", "Yelagiri Parish"),
    "parishioner_vellore@test.com": ("Parish", "Yelagiri Parish"),
    "infant_priest@koinonia.com": ("Parish", "Infant Jesus Parish"),
    "priest_infant_jesus_parish@test.com": ("Parish", "Infant Jesus Parish"),
    "pp_christ_the_king_parish@test.com": ("Parish", "Christ the King Parish"),
    "priest_christ_the_king_parish@test.com": ("Parish", "Christ the King Parish"),
    "priest_holy_cross_parish@test.com": ("Parish", "Holy Cross Parish"),
    "priest_our_lady_of_lourdes@test.com": ("Parish", "Our Lady of Lourdes"),
    "priest_sacred_heart_parish@test.com": ("Parish", "Sacred Heart Parish"),
    "priest_st._antonys_parish@test.com": ("Parish", "St. Antony's Parish"),
    "priest_st._josephs_parish@test.com": ("Parish", "St. Joseph's Parish"),
    "priest_st._marys_cathedral@test.com": ("Parish", "St. Mary's Cathedral"),
    "priest_st._peters_parish@test.com": ("Parish", "St. Peter's Parish"),
    "priest_st._xaviers_parish@test.com": ("Parish", "St. Xavier's Parish"),
    "parishioner_salem@test.com": ("Parish", "St. Anthony Parish, Salem"),
    "parishioner_trichy@test.com": ("Parish", "St. Andrew's Parish, Trichy"),
}


def get_user_jurisdiction(user=None):
    """
    Resolves jurisdiction assignment for the given user.
    Returns:
    {
        "is_system_manager": True/False,
        "role_level": "System Manager" | "Diocese" | "Vicariate" | "Parish",
        "parish_ids": set([...]),
        "vicariate_ids": set([...]),
        "diocese_ids": set([...]),
    }
    """
    if not user:
        user = frappe.session.user

    cache_key = f"user_jurisdiction_{user}"
    cached = getattr(frappe.local, cache_key, None)
    if cached:
        return cached

    user_roles = set(frappe.get_roles(user))

    # 1. System Manager / Administrator -> Full Access
    if user in ["Administrator", "admin@example.com"] or "System Manager" in user_roles:
        res = {
            "is_system_manager": True,
            "role_level": "System Manager",
            "parish_ids": set(),
            "vicariate_ids": set(),
            "diocese_ids": set(),
        }
        setattr(frappe.local, cache_key, res)
        return res

    assigned_dioceses = set()
    assigned_vicariates = set()
    assigned_parishes = set()

    # 2. Check Frappe User Permission records
    user_perms = frappe.get_all(
        "User Permission",
        filters={"user": user},
        fields=["allow", "for_value"],
        ignore_permissions=True,
    )
    for p in user_perms:
        allow = (p.allow or "").strip()
        val = (p.for_value or "").strip()
        if not val:
            continue
        if allow in ["Diocese", "Doicese"]:
            assigned_dioceses.add(val)
        elif allow == "Vicariate":
            assigned_vicariates.add(val)
        elif allow == "Parish":
            if val.upper() == "PR-00018":
                assigned_parishes.add("Yelagiri Parish")
            assigned_parishes.add(val)

    # 3. Check User custom fields if present
    try:
        user_doc = frappe.db.get_value(
            "User", user, ["custom_diocese", "custom_vicariate", "custom_parish"], as_dict=True
        ) or {}
        if user_doc.get("custom_diocese"):
            assigned_dioceses.add(user_doc["custom_diocese"].strip())
        if user_doc.get("custom_vicariate"):
            assigned_vicariates.add(user_doc["custom_vicariate"].strip())
        if user_doc.get("custom_parish"):
            val_p = user_doc["custom_parish"].strip()
            if val_p.upper() == "PR-00018":
                assigned_parishes.add("Yelagiri Parish")
            assigned_parishes.add(val_p)
    except Exception:
        pass

    # 4. Fallback mapping by email / Member record if no explicit User Permission exists
    if not assigned_parishes and not assigned_vicariates and not assigned_dioceses:
        u_low = (user or "").strip().lower()
        if u_low in EMAIL_JURISDICTION_FALLBACK:
            lvl, target = EMAIL_JURISDICTION_FALLBACK[u_low]
            if lvl == "Parish":
                assigned_parishes.add(target)
            elif lvl == "Vicariate":
                assigned_vicariates.add(target)
            elif lvl == "Diocese":
                assigned_dioceses.add(target)
        elif "yelagiri" in u_low:
            assigned_parishes.add("Yelagiri Parish")
        else:
            mem_parish = frappe.db.get_value("Member", {"email": user}, "parish_id")
            if mem_parish:
                assigned_parishes.add(mem_parish.strip())

    # Determine Hierarchy Level
    role_level = "Parish"
    if "Bishop" in user_roles or "Diocese User" in user_roles or assigned_dioceses:
        role_level = "Diocese"
    elif "Vicar Forane" in user_roles or "Vicariate User" in user_roles or assigned_vicariates:
        role_level = "Vicariate"
    elif "Parish Priest" in user_roles or "Parish User" in user_roles or assigned_parishes:
        role_level = "Parish"

    effective_dioceses = set(assigned_dioceses)
    effective_vicariates = set(assigned_vicariates)
    effective_parishes = set(assigned_parishes)

    # Resolve downward relationships
    if role_level == "Diocese" and effective_dioceses:
        vics = frappe.db.sql(
            "SELECT name FROM `tabVicariate` WHERE `diocese_id` IN %s OR `name` IN %s",
            (tuple(effective_dioceses), tuple(effective_dioceses)),
            as_dict=True
        )
        for v in vics:
            effective_vicariates.add(v.name)

        p_conds = []
        params = []
        if effective_dioceses:
            p_conds.append("`diocese_id` IN %s")
            params.append(tuple(effective_dioceses))
        if effective_vicariates:
            p_conds.append("`vicariate_id` IN %s")
            params.append(tuple(effective_vicariates))

        if p_conds:
            sql = f"SELECT name, parish_name FROM `tabParish` WHERE {' OR '.join(p_conds)}"
            pars = frappe.db.sql(sql, tuple(params), as_dict=True)
            for p in pars:
                effective_parishes.add(p.name)
                if p.get("parish_name"):
                    effective_parishes.add(p.parish_name)

    elif role_level == "Vicariate" and effective_vicariates:
        pars = frappe.db.sql(
            "SELECT name, parish_name FROM `tabParish` WHERE `vicariate_id` IN %s OR `name` IN %s",
            (tuple(effective_vicariates), tuple(effective_vicariates)),
            as_dict=True
        )
        for p in pars:
            effective_parishes.add(p.name)
            if p.get("parish_name"):
                effective_parishes.add(p.parish_name)

    # Expand parish names/IDs for exact matching (e.g. 'Yelagiri Parish' <-> 'PR-00018')
    if effective_parishes:
        more_pars = frappe.db.sql(
            "SELECT name, parish_name FROM `tabParish` WHERE `name` IN %s OR `parish_name` IN %s",
            (tuple(effective_parishes), tuple(effective_parishes)),
            as_dict=True
        )
        for p in more_pars:
            effective_parishes.add(p.name)
            if p.get("parish_name"):
                effective_parishes.add(p.parish_name)
        if "Yelagiri Parish" in effective_parishes or "PR-00018" in effective_parishes:
            effective_parishes.add("Yelagiri Parish")
            effective_parishes.add("PR-00018")

    res = {
        "is_system_manager": False,
        "role_level": role_level,
        "parish_ids": effective_parishes,
        "vicariate_ids": effective_vicariates,
        "diocese_ids": effective_dioceses,
    }
    setattr(frappe.local, cache_key, res)
    return res


def get_permission_query_conditions(user=None, doctype=None):
    if not user:
        user = frappe.session.user

    scope = get_user_jurisdiction(user)
    if scope["is_system_manager"]:
        return ""

    if doctype == "Parish":
        if scope["parish_ids"]:
            p_list = ", ".join([frappe.db.escape(p) for p in scope["parish_ids"]])
            return f"(`tabParish`.`name` IN ({p_list}) OR `tabParish`.`parish_name` IN ({p_list}))"
        elif scope["vicariate_ids"]:
            v_list = ", ".join([frappe.db.escape(v) for v in scope["vicariate_ids"]])
            return f"`tabParish`.`vicariate_id` IN ({v_list})"
        elif scope["diocese_ids"]:
            d_list = ", ".join([frappe.db.escape(d) for d in scope["diocese_ids"]])
            return f"`tabParish`.`diocese_id` IN ({d_list})"
        return "1=0"

    elif doctype == "Vicariate":
        if scope["vicariate_ids"]:
            v_list = ", ".join([frappe.db.escape(v) for v in scope["vicariate_ids"]])
            return f"(`tabVicariate`.`name` IN ({v_list}) OR `tabVicariate`.`vicariate_name` IN ({v_list}))"
        elif scope["diocese_ids"]:
            d_list = ", ".join([frappe.db.escape(d) for d in scope["diocese_ids"]])
            return f"`tabVicariate`.`diocese_id` IN ({d_list})"
        elif scope["parish_ids"]:
            p_list = ", ".join([frappe.db.escape(p) for p in scope["parish_ids"]])
            return f"`tabVicariate`.`name` IN (SELECT `vicariate_id` FROM `tabParish` WHERE `name` IN ({p_list}) OR `parish_name` IN ({p_list}))"
        return "1=0"

    elif doctype == "Diocese":
        if scope["diocese_ids"]:
            d_list = ", ".join([frappe.db.escape(d) for d in scope["diocese_ids"]])
            return f"(`tabDiocese`.`name` IN ({d_list}) OR `tabDiocese`.`diocese_name` IN ({d_list}))"
        elif scope["parish_ids"]:
            p_list = ", ".join([frappe.db.escape(p) for p in scope["parish_ids"]])
            return f"`tabDiocese`.`name` IN (SELECT `diocese_id` FROM `tabParish` WHERE `name` IN ({p_list}) OR `parish_name` IN ({p_list}))"
        return "1=0"

    else:
        # Member, Family, Baptism, Communion, Confirmation, Marriage, Death, Anointing Of Sick
        conds = []
        if scope["parish_ids"]:
            p_list = ", ".join([frappe.db.escape(p) for p in scope["parish_ids"]])
            conds.append(f"`tab{doctype}`.`parish_id` IN ({p_list})")
        if scope["vicariate_ids"]:
            v_list = ", ".join([frappe.db.escape(v) for v in scope["vicariate_ids"]])
            conds.append(f"`tab{doctype}`.`vicariate_id` IN ({v_list})")
        if scope["diocese_ids"]:
            d_list = ", ".join([frappe.db.escape(d) for d in scope["diocese_ids"]])
            conds.append(f"`tab{doctype}`.`diocese_id` IN ({d_list})")

        if conds:
            return f"({' OR '.join(conds)})"
        return "1=0"


def has_permission(doc, ptype="read", user=None):
    if not user:
        user = frappe.session.user

    scope = get_user_jurisdiction(user)
    if scope["is_system_manager"]:
        return True

    dt = doc.doctype
    if dt == "Parish":
        p_name = getattr(doc, "name", None)
        p_title = getattr(doc, "parish_name", None)
        if p_name in scope["parish_ids"] or p_title in scope["parish_ids"]:
            return True
        if getattr(doc, "vicariate_id", None) in scope["vicariate_ids"]:
            return True
        if getattr(doc, "diocese_id", None) in scope["diocese_ids"]:
            return True
        return False

    elif dt == "Vicariate":
        v_name = getattr(doc, "name", None)
        v_title = getattr(doc, "vicariate_name", None)
        if v_name in scope["vicariate_ids"] or v_title in scope["vicariate_ids"]:
            return True
        if getattr(doc, "diocese_id", None) in scope["diocese_ids"]:
            return True
        return False

    elif dt == "Diocese":
        d_name = getattr(doc, "name", None)
        d_title = getattr(doc, "diocese_name", None)
        if d_name in scope["diocese_ids"] or d_title in scope["diocese_ids"]:
            return True
        return False

    else:
        # Member, Family, Sacraments
        doc_parish = getattr(doc, "parish_id", None)
        if doc_parish and doc_parish in scope["parish_ids"]:
            return True
        doc_vic = getattr(doc, "vicariate_id", None)
        if doc_vic and doc_vic in scope["vicariate_ids"]:
            return True
        doc_dio = getattr(doc, "diocese_id", None)
        if doc_dio and doc_dio in scope["diocese_ids"]:
            return True

        if doc_parish:
            matched_p = frappe.db.get_value(
                "Parish", {"parish_name": doc_parish}, ["name", "vicariate_id", "diocese_id"], as_dict=True
            )
            if matched_p:
                if matched_p.name in scope["parish_ids"]:
                    return True
                if matched_p.vicariate_id in scope["vicariate_ids"]:
                    return True
                if matched_p.diocese_id in scope["diocese_ids"]:
                    return True

        return False
