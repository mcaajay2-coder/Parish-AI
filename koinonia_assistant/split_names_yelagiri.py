import frappe

PARISH_NAME = "Yelagiri Parish"

def sample_check():
    frappe.set_user("Administrator")
    print(f"=== Sample Member Records for {PARISH_NAME} ===")
    res = frappe.db.sql("SELECT name, first_name, middle_name, last_name FROM `tabMember` WHERE `parish_id` = %s LIMIT 15", (PARISH_NAME,), as_dict=True)
    for r in res:
        print(f"  ID: '{r.name}' -> First: '{r.first_name}' | Middle: '{r.middle_name}' | Last: '{r.last_name}'")

if __name__ == "__main__":
    sample_check()
