import csv
import os
import re
import frappe

PARISH_NAME = "Yelagiri Parish"
DATA_DIR = "/home/frappe/frappe-bench/apps/koinonia_assistant/koinonia_assistant/Real time data/Final Real TIme Data for Yelagiri parish"

def split_full_name(full_name):
    if not full_name or not full_name.strip():
        return "", "", ""
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0].rstrip('.'), "", ""
    elif len(parts) == 2:
        p2 = parts[1].rstrip('.')
        if len(p2) <= 3:
            return parts[0].rstrip('.'), "", p2
        else:
            return parts[0].rstrip('.'), p2.rstrip('.'), ""
    else:
        first_name = parts[0].rstrip('.')
        last_name = parts[-1].rstrip('.')
        middle_name = " ".join(parts[1:-1]).rstrip('.')
        return first_name, middle_name, last_name

def run_ingestion():
    frappe.set_user("Administrator")
    print(f"=== Starting Yelagiri Ingestion (Pure Numeric Member IDs, 3-Way Name Split & Dot-less Last Names) ===")

    # 1. SCOPED PURGE OF OLD YELAGIRI PARISH DATA
    print("--- Purging existing records for Yelagiri Parish ---")
    doctypes = ["Member", "Family", "Baptism", "Communion", "Confirmation", "Marriage", "Death"]
    for dt in doctypes:
        table_name = f"tab{dt}"
        count_before = frappe.db.count(dt, filters={"parish_id": PARISH_NAME})
        frappe.db.sql(f"DELETE FROM `{table_name}` WHERE `parish_id` = %s", (PARISH_NAME,))
        print(f"Purged {count_before} records from {dt} for {PARISH_NAME}")
    frappe.db.commit()

    # 2. READ live_family_data.csv
    family_file = os.path.join(DATA_DIR, "live_family_data.csv")
    if not os.path.exists(family_file):
        raise FileNotFoundError(f"File not found: {family_file}")

    families_inserted = 0
    members_inserted = 0
    
    member_key_map = {}
    all_members = []

    family_rows_by_id = {}
    with open(family_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam_id = row["family_id"].strip()
            if not fam_id:
                continue
            if fam_id not in family_rows_by_id:
                family_rows_by_id[fam_id] = []
            family_rows_by_id[fam_id].append(row)

    print(f"Found {len(family_rows_by_id)} unique families in live_family_data.csv")

    for fam_id, member_rows in family_rows_by_id.items():
        first_row = member_rows[0]
        card_no = first_row.get("family_card_number", "").strip() or None
        reg_no = first_row.get("register_number", "").strip() or None
        fam_name = first_row.get("family_name", "").strip() or card_no or fam_id

        # Insert tabFamily doc
        fam_doc = frappe.get_doc({
            "doctype": "Family",
            "name": fam_doc_name,
            "family_card_number": card_no,
            "family_register_number": reg_no,
            "reference": fam_name,
            "parish_bcc_id": first_row.get("basic_christian_community", "").strip(),
            "zone_id": first_row.get("zone_id", "").strip(),
            "lang_community_id": first_row.get("language_community_id", "").strip(),
            "rite_id": first_row.get("rite_id", "").strip(),
            "status": first_row.get("status", "Approved").strip(),
            "active": 1 if first_row.get("active", "").lower() == 't' else 0,
            "active_in_parish": first_row.get("active_in_parish", "t").strip(),
            "register_date": first_row.get("registration_date", "").strip() or None,
            "house_ownership": first_row.get("house_ownership", "").strip(),
            "rent_amt": float(first_row.get("rent_amount")) if first_row.get("rent_amount") and first_row.get("rent_amount").replace('.', '', 1).isdigit() else 0,
            "street": first_row.get("address_line_1", "").strip(),
            "city": first_row.get("city", "").strip(),
            "district_id": first_row.get("district_id", "").strip(),
            "state_id": first_row.get("state_id", "").strip(),
            "country_id": first_row.get("country_id", "").strip(),
            "zip": first_row.get("zip", "").strip(),
            "phone": first_row.get("phone", "").strip(),
            "mobile": first_row.get("mobile", "").strip(),
            "email": first_row.get("email", "").strip(),
            "is_civil_marriage": first_row.get("civil_marriage", "").strip(),
            "is_church_marriage": first_row.get("church_marriage", "").strip(),
            "comment": first_row.get("comment", "").strip(),
            "diocese_id": first_row.get("diocese_id", "").strip(),
            "vicariate_id": first_row.get("vicariate_id", "").strip(),
            "parish_id": PARISH_NAME
        })
        fam_doc.insert(ignore_permissions=True)
        families_inserted += 1

        # Insert tabMember docs for each member in family
        for idx, m_row in enumerate(member_rows, start=1):
            mem_name = m_row.get("member_name", "").strip()
            if not mem_name:
                continue

            fn, mn, ln = split_full_name(mem_name)

            mem_key = f"{fam_id}{idx}"
            mem_doc = frappe.get_doc({
                "doctype": "Member",
                "name": mem_key,
                "first_name": fn,
                "middle_name": mn,
                "last_name": ln,
                "family_id": fam_doc_name,
                "parish_id": PARISH_NAME,
                "gender": m_row.get("gender", "").strip(),
                "mobile": m_row.get("member_mobile", "").strip(),
                "email": m_row.get("member_email", "").strip(),
                "status": m_row.get("status", "Approved").strip(),
                "active": 1 if m_row.get("active", "").lower() == 't' else 0,
                "diocese_id": m_row.get("diocese_id", "").strip(),
                "vicariate_id": m_row.get("vicariate_id", "").strip()
            })
            mem_doc.insert(ignore_permissions=True)
            members_inserted += 1

            clean_name = re.sub(r'[^a-z0-9]', '', mem_name.lower())
            member_key_map[(str(fam_id), clean_name)] = mem_key
            if card_no:
                member_key_map[(card_no.lower(), clean_name)] = mem_key
            all_members.append((clean_name, mem_key))

    frappe.db.commit()
    print(f"Successfully inserted {families_inserted} Family records and {members_inserted} Member records!")

    def update_member_sacrament_date(fam_card_or_id, full_name, field_name, date_val):
        if not full_name or not date_val:
            return
        c_name = re.sub(r'[^a-z0-9]', '', full_name.strip().lower())
        c_fam = str(fam_card_or_id).strip().lower() if fam_card_or_id else ""
        
        target_mem_key = None
        if c_fam:
            target_mem_key = member_key_map.get((c_fam, c_name))
        
        if not target_mem_key:
            for m_clean, m_key in all_members:
                if c_name in m_clean or m_clean in c_name:
                    target_mem_key = m_key
                    break

        if target_mem_key:
            frappe.db.set_value("Member", target_mem_key, field_name, date_val)

    # 3. INGEST BAPTISM
    bap_file = os.path.join(DATA_DIR, "member_baptism.csv")
    bap_count = 0
    if os.path.exists(bap_file):
        with open(bap_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                full_name = row.get("full_name", "").strip() or row.get("name", "").strip()
                fam_card = row.get("family_card_no", "").strip() or row.get("family_id", "").strip()
                bap_id = row.get("id", "").strip() or str(idx)
                b_date = row.get("bapt_date", "").strip() or None
                
                fn, mn, ln = split_full_name(full_name)

                doc = frappe.get_doc({
                    "doctype": "Baptism",
                    "name": f"BAP-YLG-{bap_id}",
                    "first_name": fn,
                    "middle_name": mn,
                    "last_name": ln,
                    "bapt_register_ref": row.get("bapt_register_ref", "").strip(),
                    "bapt_date": b_date,
                    "bapt_place": row.get("bapt_place", "").strip(),
                    "bapt_minister": row.get("bapt_minister", "").strip(),
                    "parish_priest": row.get("parish_priest", "").strip(),
                    "family_card_no": fam_card,
                    "father_name": row.get("father_name", "").strip(),
                    "mother_name": row.get("mother_name", "").strip(),
                    "bapt_god_father": row.get("bapt_god_father", "").strip(),
                    "bapt_god_mother": row.get("bapt_god_mother", "").strip(),
                    "gender": row.get("gender", "").strip(),
                    "dob": row.get("dob", "").strip() or None,
                    "parish_id": PARISH_NAME
                })
                doc.insert(ignore_permissions=True)
                bap_count += 1

                if b_date:
                    update_member_sacrament_date(fam_card, full_name, "bapt_date", b_date)
    print(f"Inserted {bap_count} Baptism records!")

    # 4. INGEST COMMUNION
    com_file = os.path.join(DATA_DIR, "member_communion.csv")
    com_count = 0
    if os.path.exists(com_file):
        with open(com_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                full_name = row.get("full_name", "").strip() or row.get("name", "").strip()
                fam_card = row.get("family_card_no", "").strip()
                com_id = row.get("id", "").strip() or str(idx)
                c_date = row.get("fhc_date", "").strip() or None
                
                fn, mn, ln = split_full_name(full_name)

                doc = frappe.get_doc({
                    "doctype": "Communion",
                    "name": f"COM-YLG-{com_id}",
                    "first_name": fn,
                    "middle_name": mn,
                    "last_name": ln,
                    "fhc_register_ref": row.get("fhc_register_ref", "").strip(),
                    "fhc_date": c_date,
                    "fhc_place": row.get("fhc_place", "").strip(),
                    "fhc_minister": row.get("fhc_minister", "").strip(),
                    "parish_priest": row.get("parish_priest", "").strip(),
                    "family_card_no": fam_card,
                    "father_name": row.get("father_name", "").strip(),
                    "mother_name": row.get("mother_name", "").strip(),
                    "gender": row.get("gender", "").strip(),
                    "dob": row.get("dob", "").strip() or None,
                    "parish_id": PARISH_NAME
                })
                doc.insert(ignore_permissions=True)
                com_count += 1

                if c_date:
                    update_member_sacrament_date(fam_card, full_name, "fhc_date", c_date)
    print(f"Inserted {com_count} Communion records!")

    # 5. INGEST CONFIRMATION
    cnf_file = os.path.join(DATA_DIR, "member_confirmation.csv")
    cnf_count = 0
    if os.path.exists(cnf_file):
        with open(cnf_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                full_name = row.get("full_name", "").strip() or row.get("name", "").strip()
                fam_card = row.get("family_card_no", "").strip()
                cnf_id = row.get("id", "").strip() or str(idx)
                cnf_d = row.get("cnf_date", "").strip() or None
                
                fn, mn, ln = split_full_name(full_name)

                doc = frappe.get_doc({
                    "doctype": "Confirmation",
                    "name": f"CNF-YLG-{cnf_id}",
                    "first_name": fn,
                    "middle_name": mn,
                    "last_name": ln,
                    "cnf_register_ref": row.get("cnf_register_ref", "").strip(),
                    "cnf_date": cnf_d,
                    "cnf_place": row.get("cnf_place", "").strip(),
                    "cnf_minister": row.get("cnf_minister", "").strip(),
                    "parish_priest": row.get("parish_priest", "").strip(),
                    "family_card_no": fam_card,
                    "father_name": row.get("father_name", "").strip(),
                    "mother_name": row.get("mother_name", "").strip(),
                    "gender": row.get("gender", "").strip(),
                    "dob": row.get("dob", "").strip() or None,
                    "parish_id": PARISH_NAME
                })
                doc.insert(ignore_permissions=True)
                cnf_count += 1

                if cnf_d:
                    update_member_sacrament_date(fam_card, full_name, "cnf_date", cnf_d)
    print(f"Inserted {cnf_count} Confirmation records!")

    # 6. INGEST MARRIAGE
    mrg_file = os.path.join(DATA_DIR, "member_marriage.csv")
    mrg_count = 0
    if os.path.exists(mrg_file):
        with open(mrg_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                mrg_id = row.get("id", "").strip() or str(idx)
                m_date = row.get("mrg_date", "").strip() or None
                bg_full = row.get("bridegroom_full_name", "").strip() or row.get("bridegroom_name", "").strip()
                b_full = row.get("bride_full_name", "").strip() or row.get("bride_name", "").strip()
                
                bg_fn, bg_mn, bg_ln = split_full_name(bg_full)
                b_fn, b_mn, b_ln = split_full_name(b_full)

                doc = frappe.get_doc({
                    "doctype": "Marriage",
                    "name": f"MRG-YLG-{mrg_id}",
                    "bridegroom_name": bg_fn,
                    "bridegroom_middle_name": bg_mn,
                    "bridegroom_last_name": bg_ln,
                    "bride_name": b_fn,
                    "bride_middle_name": b_mn,
                    "bride_last_name": b_ln,
                    "mrg_register_ref": row.get("mrg_register_ref", "").strip(),
                    "mrg_date": m_date,
                    "mrg_place": row.get("mrg_place", "").strip(),
                    "mrg_minister": row.get("mrg_minister", "").strip(),
                    "parish_priest": row.get("parish_priest", "").strip(),
                    "family_card_no": row.get("family_card_no", "").strip(),
                    "parish_id": PARISH_NAME
                })
                doc.insert(ignore_permissions=True)
                mrg_count += 1
                
                fam_card = row.get("family_card_no", "").strip()
                if m_date:
                    update_member_sacrament_date(fam_card, bg_full, "mrg_date", m_date)
                    update_member_sacrament_date(fam_card, b_full, "mrg_date", m_date)
    print(f"Inserted {mrg_count} Marriage records!")

    # 7. INGEST DEATH
    dth_file = os.path.join(DATA_DIR, "member_death.csv")
    dth_count = 0
    if os.path.exists(dth_file):
        with open(dth_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                dth_id = row.get("id", "").strip() or str(idx)
                full_name = row.get("full_name", "").strip() or row.get("name", "").strip()
                fn, mn, ln = split_full_name(full_name)

                doc = frappe.get_doc({
                    "doctype": "Death",
                    "name": f"DTH-YLG-{dth_id}",
                    "first_name": fn,
                    "middle_name": mn,
                    "last_name": ln,
                    "death_register_ref": row.get("death_register_ref", "").strip(),
                    "death_date": row.get("death_date", "").strip() or None,
                    "death_place": row.get("death_place", "").strip(),
                    "death_cause": row.get("death_cause", "").strip(),
                    "burial_date": row.get("burial_date", "").strip() or None,
                    "burial_place": row.get("burial_place", "").strip(),
                    "burial_minister": row.get("burial_minister", "").strip(),
                    "parish_priest": row.get("parish_priest", "").strip(),
                    "family_card_no": row.get("family_card_no", "").strip(),
                    "parish_id": PARISH_NAME
                })
                doc.insert(ignore_permissions=True)
                dth_count += 1
    print(f"Inserted {dth_count} Death records!")

    frappe.db.commit()
    print("=== Yelagiri Parish Ingestion Completed Successfully! ===")
