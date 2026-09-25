import frappe

PARISH_NAME = "Yelagiri Parish"

def run_dot_strip():
    frappe.set_user("Administrator")
    print(f"=== Stripping trailing dots from last_name for Parish: {PARISH_NAME} ===")

    # 1. Update tabMember
    members = frappe.db.sql(f"SELECT name, last_name FROM `tabMember` WHERE `parish_id` = %s AND `last_name` LIKE '%%.'", (PARISH_NAME,), as_dict=True)
    mem_updated = 0
    for m in members:
        ln = (m.last_name or "").strip()
        new_ln = ln.rstrip('.')
        if new_ln != ln:
            frappe.db.sql("UPDATE `tabMember` SET `last_name` = %s WHERE `name` = %s", (new_ln, m.name))
            mem_updated += 1
    print(f"Updated {mem_updated} Member last_names!")

    # 2. Update tabBaptism
    baptisms = frappe.db.sql(f"SELECT name, last_name FROM `tabBaptism` WHERE `parish_id` = %s AND `last_name` LIKE '%%.'", (PARISH_NAME,), as_dict=True)
    bap_updated = 0
    for b in baptisms:
        ln = (b.last_name or "").strip()
        new_ln = ln.rstrip('.')
        if new_ln != ln:
            frappe.db.sql("UPDATE `tabBaptism` SET `last_name` = %s WHERE `name` = %s", (new_ln, b.name))
            bap_updated += 1
    print(f"Updated {bap_updated} Baptism last_names!")

    # 3. Update tabCommunion
    communions = frappe.db.sql(f"SELECT name, last_name FROM `tabCommunion` WHERE `parish_id` = %s AND `last_name` LIKE '%%.'", (PARISH_NAME,), as_dict=True)
    com_updated = 0
    for c in communions:
        ln = (c.last_name or "").strip()
        new_ln = ln.rstrip('.')
        if new_ln != ln:
            frappe.db.sql("UPDATE `tabCommunion` SET `last_name` = %s WHERE `name` = %s", (new_ln, c.name))
            com_updated += 1
    print(f"Updated {com_updated} Communion last_names!")

    # 4. Update tabConfirmation
    confirmations = frappe.db.sql(f"SELECT name, last_name FROM `tabConfirmation` WHERE `parish_id` = %s AND `last_name` LIKE '%%.'", (PARISH_NAME,), as_dict=True)
    cnf_updated = 0
    for c in confirmations:
        ln = (c.last_name or "").strip()
        new_ln = ln.rstrip('.')
        if new_ln != ln:
            frappe.db.sql("UPDATE `tabConfirmation` SET `last_name` = %s WHERE `name` = %s", (new_ln, c.name))
            cnf_updated += 1
    print(f"Updated {cnf_updated} Confirmation last_names!")

    # 5. Update tabMarriage
    marriages = frappe.db.sql(f"SELECT name, bridegroom_last_name, bride_last_name FROM `tabMarriage` WHERE `parish_id` = %s", (PARISH_NAME,), as_dict=True)
    mrg_updated = 0
    for m in marriages:
        bg_ln = (m.bridegroom_last_name or "").strip()
        b_ln = (m.bride_last_name or "").strip()

        updates = []
        params = []
        if bg_ln.endswith('.'):
            new_bg_ln = bg_ln.rstrip('.')
            updates.append("`bridegroom_last_name` = %s")
            params.append(new_bg_ln)

        if b_ln.endswith('.'):
            new_b_ln = b_ln.rstrip('.')
            updates.append("`bride_last_name` = %s")
            params.append(new_b_ln)

        if updates:
            sql = f"UPDATE `tabMarriage` SET {', '.join(updates)} WHERE `name` = %s"
            params.append(m.name)
            frappe.db.sql(sql, tuple(params))
            mrg_updated += 1
    print(f"Updated {mrg_updated} Marriage last_names!")

    # 6. Update tabDeath
    deaths = frappe.db.sql(f"SELECT name, last_name FROM `tabDeath` WHERE `parish_id` = %s AND `last_name` LIKE '%%.'", (PARISH_NAME,), as_dict=True)
    dth_updated = 0
    for d in deaths:
        ln = (d.last_name or "").strip()
        new_ln = ln.rstrip('.')
        if new_ln != ln:
            frappe.db.sql("UPDATE `tabDeath` SET `last_name` = %s WHERE `name` = %s", (new_ln, d.name))
            dth_updated += 1
    print(f"Updated {dth_updated} Death last_names!")

    frappe.db.commit()
    print("=== Stripping Trailing Dots Completed Successfully! ===")

if __name__ == "__main__":
    run_dot_strip()
