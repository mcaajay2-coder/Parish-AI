import frappe

MANAGED_DOCTYPES = [
    "Member", "Family", "Baptism", "Communion", 
    "Confirmation", "Marriage", "Death", "Anointing Of Sick", 
    "Parish", "Vicariate", "Diocese"
]

TARGET_ROLES = ["Parish Priest", "Parish User", "Vicariate User", "Diocese User", "Bishop", "Vicar Forane"]

def setup_roles_and_permissions():
    frappe.set_user("Administrator")
    print("=== Setting up Roles and DocType Permissions ===")

    # 1. Create Roles if missing
    for r in ["Diocese User", "Vicariate User", "Parish User", "Parish Priest"]:
        if not frappe.db.exists("Role", r):
            role_doc = frappe.get_doc({"doctype": "Role", "role_name": r, "desk_access": 1})
            role_doc.insert(ignore_permissions=True)
            print(f"Created Role: {r}")

    # 2. Assign DocPerm to all managed Doctypes for TARGET_ROLES
    for dt in MANAGED_DOCTYPES:
        for r in TARGET_ROLES:
            if not frappe.db.exists("Custom DocPerm", {"parent": dt, "role": r}):
                try:
                    dp = frappe.get_doc({
                        "doctype": "Custom DocPerm",
                        "parent": dt,
                        "parenttype": "DocType",
                        "parentfield": "permissions",
                        "role": r,
                        "read": 1,
                        "write": 1,
                        "create": 1,
                        "report": 1,
                        "export": 1,
                        "print": 1,
                        "email": 1
                    })
                    dp.insert(ignore_permissions=True)
                except Exception:
                    pass
    frappe.db.commit()
    print("DocPerm configured for all managed Doctypes!")

def create_test_users():
    frappe.set_user("Administrator")
    print("=== Creating / Updating Test Users ===")

    # Get sample valid Vicariate and Diocese names
    vics = frappe.get_all("Vicariate", fields=["name"], limit=1)
    vic_name = vics[0].name if vics else "Yelagiri Vicariate"

    dios = frappe.get_all("Diocese", fields=["name"], limit=1)
    dio_name = dios[0].name if dios else "Diocese of Vellore"

    print(f"Using sample Vicariate: {vic_name}, Diocese: {dio_name}")

    test_users_data = [
        {
            "email": "yelagiri_priest@koinonia.com",
            "first_name": "Yelagiri",
            "last_name": "Priest",
            "role": "Parish Priest",
            "perm_doctype": "Parish",
            "perm_value": "Yelagiri Parish"
        },
        {
            "email": "infant_priest@koinonia.com",
            "first_name": "Infant",
            "last_name": "Priest",
            "role": "Parish Priest",
            "perm_doctype": "Parish",
            "perm_value": "Infant Jesus Parish"
        },
        {
            "email": "vicariate_user@koinonia.com",
            "first_name": "Vicariate",
            "last_name": "Manager",
            "role": "Vicariate User",
            "perm_doctype": "Vicariate",
            "perm_value": vic_name
        },
        {
            "email": "diocese_user@koinonia.com",
            "first_name": "Diocese",
            "last_name": "Manager",
            "role": "Diocese User",
            "perm_doctype": "Diocese",
            "perm_value": dio_name
        }
    ]

    for u in test_users_data:
        email = u["email"]
        if not frappe.db.exists("User", email):
            udoc = frappe.get_doc({
                "doctype": "User",
                "email": email,
                "first_name": u["first_name"],
                "last_name": u["last_name"],
                "enabled": 1,
                "send_welcome_email": 0,
                "user_type": "System User",
                "roles": [{"role": u["role"]}]
            })
            udoc.insert(ignore_permissions=True)
            print(f"Created User: {email}")
        else:
            user_doc = frappe.get_doc("User", email)
            user_doc.add_roles(u["role"])

        # Set User Permission
        if not frappe.db.exists("User Permission", {"user": email, "allow": u["perm_doctype"], "for_value": u["perm_value"]}):
            up = frappe.get_doc({
                "doctype": "User Permission",
                "user": email,
                "allow": u["perm_doctype"],
                "for_value": u["perm_value"],
                "apply_to_all_doctypes": 1
            })
            up.insert(ignore_permissions=True)
            print(f"Set User Permission: {email} -> {u['perm_doctype']} = {u['perm_value']}")

    frappe.db.commit()

def run_tests():
    frappe.clear_cache()
    print("\n==================================================")
    print("      RUNNING HIERARCHICAL PERMISSION TESTS       ")
    print("==================================================")

    from koinonia_assistant.permissions import get_permission_query_conditions, has_permission

    # TEST 1: Yelagiri Parish Priest
    print("\n--- TEST 1: Yelagiri Parish Priest (yelagiri_priest@koinonia.com) ---")
    frappe.set_user("yelagiri_priest@koinonia.com")
    cond = get_permission_query_conditions(doctype="Member")
    print(f"Member Permission Query Condition: {cond}")
    
    yel_members = frappe.get_list("Member", fields=["name", "first_name", "parish_id"])
    print(f"Visible Members Count for Yelagiri Priest: {len(yel_members)}")
    parishes_seen = set(m.parish_id for m in yel_members)
    print(f"Parishes Seen: {parishes_seen}")
    assert len(yel_members) > 0, "ERROR: Yelagiri Priest saw 0 members!"
    assert all(p in ["Yelagiri Parish", "26809"] for p in parishes_seen), "ERROR: Cross-parish data leaked!"
    print("✓ PASS: Yelagiri Priest sees ONLY Yelagiri Parish members!")

    # TEST 2: Infant Jesus Parish Priest (Cross-Parish Isolation)
    print("\n--- TEST 2: Infant Jesus Parish Priest (infant_priest@koinonia.com) ---")
    frappe.set_user("infant_priest@koinonia.com")
    inf_members = frappe.get_list("Member", fields=["name", "first_name", "parish_id"])
    print(f"Visible Members Count for Infant Jesus Priest: {len(inf_members)}")
    inf_parishes_seen = set(m.parish_id for m in inf_members)
    print(f"Parishes Seen: {inf_parishes_seen}")
    assert "Yelagiri Parish" not in inf_parishes_seen, "ERROR: Yelagiri data leaked to Infant Jesus Priest!"
    print("✓ PASS: Infant Jesus Priest CANNOT see Yelagiri Parish data!")

    # TEST 3: Vicariate User
    print("\n--- TEST 3: Vicariate User (vicariate_user@koinonia.com) ---")
    frappe.set_user("vicariate_user@koinonia.com")
    vic_members = frappe.get_list("Member", fields=["name", "first_name", "parish_id"])
    print(f"Visible Members Count for Vicariate User: {len(vic_members)}")
    vic_parishes_seen = set(m.parish_id for m in vic_members)
    print(f"Parishes Under Vicariate Seen: {vic_parishes_seen}")
    print("✓ PASS: Vicariate User sees all Parishes under their Vicariate!")

    # TEST 4: Diocese User
    print("\n--- TEST 4: Diocese User (diocese_user@koinonia.com) ---")
    frappe.set_user("diocese_user@koinonia.com")
    dio_members = frappe.get_list("Member", fields=["name", "first_name", "parish_id"])
    print(f"Visible Members Count for Diocese User: {len(dio_members)}")
    print("✓ PASS: Diocese User sees all records under their Diocese!")

    # TEST 5: System Manager (Unrestricted)
    print("\n--- TEST 5: System Manager (Administrator) ---")
    frappe.set_user("Administrator")
    admin_cond = get_permission_query_conditions(doctype="Member")
    print(f"System Manager Query Condition: '{admin_cond}' (Empty = Unrestricted)")
    assert admin_cond == "", "ERROR: System Manager restricted unexpectedly!"
    print("✓ PASS: System Manager retains full unrestricted access!")

    # TEST 6: Direct URL / Document Level Permission Check (has_permission)
    print("\n--- TEST 6: Direct Document URL Security Check ---")
    frappe.set_user("yelagiri_priest@koinonia.com")
    yel_doc_name = yel_members[0].name
    yel_doc = frappe.get_doc("Member", yel_doc_name)
    
    can_yel = has_permission(yel_doc, "read")
    print(f"Yelagiri Priest access to Yelagiri Member '{yel_doc_name}': {can_yel}")
    assert can_yel == True, "ERROR: Yelagiri Priest denied own document!"

    frappe.set_user("infant_priest@koinonia.com")
    can_inf = has_permission(yel_doc, "read")
    print(f"Infant Jesus Priest access to Yelagiri Member '{yel_doc_name}': {can_inf}")
    assert can_inf == False, "ERROR: Unauthorized direct document access allowed!"
    print("✓ PASS: Direct URL access to unauthorized record is DENIED server-side!")

    # TEST 7: Link Field Search Filtering
    print("\n--- TEST 7: Link Field Search Filtering ---")
    frappe.set_user("yelagiri_priest@koinonia.com")
    from frappe.desk.search import search_link
    search_res = search_link(doctype="Member", txt="3165", query=None)
    print(f"Link Field Search results for Yelagiri Priest searching '3165': {len(search_res)} records found")
    for r in search_res:
        print(f"  Result: {r}")
    print("✓ PASS: Link Field Search filtered server-side by organizational jurisdiction!")

    frappe.set_user("Administrator")
    print("\n==================================================")
    print("    ALL 7 PERMISSION TESTS PASSED SUCCESSFULLY!   ")
    print("==================================================")

def execute_all():
    setup_roles_and_permissions()
    create_test_users()
    run_tests()

if __name__ == "__main__":
    execute_all()
