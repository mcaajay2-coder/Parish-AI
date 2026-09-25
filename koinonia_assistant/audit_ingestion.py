import frappe

def audit():
    frappe.set_user("Administrator")
    print("=== AUDIT: Record Counts by Parish ===")
    
    doctypes = ["Member", "Family", "Baptism", "Communion", "Confirmation", "Marriage", "Death"]
    for dt in doctypes:
        print(f"\n--- Doctype: {dt} ---")
        results = frappe.db.sql(f"SELECT parish_id, COUNT(1) as cnt FROM `tab{dt}` GROUP BY parish_id", as_dict=True)
        for r in results:
            print(f"  Parish: {r.get('parish_id') or 'Unspecified'} -> {r.get('cnt')} records")

if __name__ == "__main__":
    audit()
