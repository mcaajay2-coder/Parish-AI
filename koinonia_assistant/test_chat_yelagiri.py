import frappe
from koinonia_assistant.rag.rag_engine import run_query
from koinonia_assistant.api import resolve_user_parish

def test_chat():
    frappe.set_user("Administrator")
    user_email = "yelagiri@koinonia.com"
    parish = resolve_user_parish(user_email)
    print(f"=== Testing RAG Chat for user: {user_email} (Resolved Parish: {parish}) ===")

    test_queries = [
        "How many total members are in Yelagiri Parish?",
        "List the members of family 3165",
        "How many baptism records are there in Yelagiri Parish?"
    ]

    for q in test_queries:
        print(f"\n[QUERY]: {q}")
        res = run_query(question=q, history=[], user_role="Parish Priest", user_parish=parish)
        print(f"[RESPONSE]:\n{res}")

if __name__ == "__main__":
    test_chat()
