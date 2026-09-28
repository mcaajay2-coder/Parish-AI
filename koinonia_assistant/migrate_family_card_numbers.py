import csv
import os
import frappe

def run():
    frappe.init("frontend", sites_path="/home/frappe/frappe-bench/sites")
    frappe.connect()

    csv_paths = [
        "/home/frappe/frappe-bench/apps/koinonia_assistant/koinonia_assistant/Real time data/Final Real TIme Data for Yelagiri parish/live_family_data.csv",
        "/tmp/live_family_data.csv"
    ]
    csv_file = None
    for p in csv_paths:
        if os.path.exists(p):
            csv_file = p
            break
            
    if not csv_file:
        raise FileNotFoundError(f"CSV file not found in any of {csv_paths}")

    print(f"Reading authoritative CSV from: {csv_file}")

    # Read unique family rows from CSV
    csv_families = {}
    with open(csv_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fid = (r.get("family_id") or "").strip()
            if not fid:
                continue
            if fid not in csv_families:
                card_no = (r.get("family_card_number") or "").strip() or None
                reg_no = (r.get("register_number") or "").strip() or None
                csv_families[fid] = {
                    "family_id": fid,
                    "family_card_number": card_no,
                    "family_register_number": reg_no,
                    "family_name": (r.get("family_name") or "").strip(),
                    "bcc": (r.get("basic_christian_community") or "").strip()
                }

    print(f"Total unique family IDs in CSV: {len(csv_families)}")

    # Fetch all DB families for Yelagiri Parish
    db_families = frappe.db.sql(
        "SELECT name, parish_id, parish_bcc_id, reference, family_card_number, family_register_number FROM tabFamily WHERE parish_id = 'Yelagiri Parish'",
        as_dict=True
    )
    total_db_families = len(db_families)
    print(f"Total Yelagiri Parish families currently in tabFamily: {total_db_families}")

    matched_by_id = 0
    matched_by_fallback = 0
    updated_card_count = 0
    updated_reg_count = 0
    unresolved_csv = []

    # Map DB families by name (primary ID)
    db_by_name = {f["name"]: f for f in db_families}

    for fid, fam_info in csv_families.items():
        db_fam = db_by_name.get(fid)
        if db_fam:
            matched_by_id += 1
        else:
            # Fallback search by reference/family_name
            fallback = frappe.db.sql(
                "SELECT name, parish_id, parish_bcc_id, reference, family_card_number, family_register_number FROM tabFamily WHERE parish_id = 'Yelagiri Parish' AND reference = %s",
                (fam_info["family_name"],),
                as_dict=True
            )
            if fallback:
                db_fam = fallback[0]
                matched_by_fallback += 1
            else:
                unresolved_csv.append(fam_info)
                continue

        card_val = fam_info["family_card_number"]
        reg_val = fam_info["family_register_number"]

        frappe.db.sql(
            """
            UPDATE `tabFamily`
            SET family_card_number = %s,
                family_register_number = %s,
                modified = NOW()
            WHERE name = %s
            """,
            (card_val, reg_val, db_fam["name"])
        )

        if card_val:
            updated_card_count += 1
        if reg_val:
            updated_reg_count += 1

    frappe.db.commit()

    print("\n=================== MIGRATION SUMMARY ===================")
    print(f"Total Yelagiri families in DB: {total_db_families}")
    print(f"Total matched by family_id: {matched_by_id}")
    print(f"Total matched by fallback: {matched_by_fallback}")
    print(f"Total updated with family_card_number: {updated_card_count}")
    print(f"Total updated with family_register_number: {updated_reg_count}")
    print(f"Unresolved CSV families (not in DB): {len(unresolved_csv)}")
    for u in unresolved_csv:
        print(f"  - CSV FID: {u['family_id']}, Name: {u['family_name']}, Card: {u['family_card_number']}, Reg: {u['family_register_number']}")

    print("\n=================== VERIFICATION OF KEY CASES ===================")
    test_fids = ["3168", "3171", "3170", "3169", "3165", "3166"]
    for tfid in test_fids:
        row = frappe.db.sql(
            "SELECT name, reference, parish_id, family_card_number, family_register_number FROM tabFamily WHERE name = %s",
            (tfid,),
            as_dict=True
        )
        if row:
            r = row[0]
            print(f"Family {r['name']} ({r['reference']}): Card No = '{r['family_card_number']}', Reg No = '{r['family_register_number']}'")
        else:
            print(f"Family {tfid} NOT found in DB")

    # Also verify member relationships
    print("\n=================== VERIFYING MEMBER RELATIONSHIPS ===================")
    mem_counts = frappe.db.sql(
        "SELECT COUNT(*) as cnt FROM tabMember WHERE family_id IN (SELECT name FROM tabFamily WHERE parish_id = 'Yelagiri Parish')",
        as_dict=True
    )[0]["cnt"]
    print(f"Total members linked to Yelagiri families: {mem_counts}")

if __name__ == "__main__":
    run()
