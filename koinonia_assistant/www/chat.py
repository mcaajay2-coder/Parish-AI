import frappe
import os

no_cache = 1

def get_context(context):
    context.no_sidebar = 1
    context.no_header = 1
    context.no_footer = 1
    
    if frappe.session.user == 'Guest':
        frappe.local.flags.redirect_to = '/login'
        raise frappe.Redirect
        
    try:
        user_doc = frappe.get_doc('User', frappe.session.user)
        context.user_fullname = user_doc.full_name
        context.user_email = user_doc.email
        context.user_first_name = user_doc.first_name or (user_doc.full_name.split()[0] if user_doc.full_name else 'User')
    except Exception:
        context.user_fullname = 'Parish User'
        context.user_email = ''
        context.user_first_name = 'User'

    user = frappe.session.user
    roles = frappe.get_roles(user) if user else []
    
    if 'Bishop' in roles or user in ['Administrator', 'admin@example.com']:
        context.user_role = 'Bishop'
        context.badge_text = 'Diocese Primary Workspace'
        context.user_parish = 'All Parishes'
        context.user_diocese = 'Trichy Diocese'
    elif 'Parish Priest' in roles:
        context.user_role = 'Parish Priest'
        parish = frappe.db.get_value('Member', {'email': user}, 'parish_id') or 'Yelagiri Parish'
        context.badge_text = parish
        context.user_parish = parish
        context.user_diocese = 'Vellore Diocese'
    else:
        context.user_role = 'Parishioner'
        parish = frappe.db.get_value('Member', {'email': user}, 'parish_id') or 'Yelagiri Parish'
        context.badge_text = parish
        context.user_parish = parish
        context.user_diocese = 'Vellore Diocese'

    try:
        context.csrf_token = frappe.sessions.get_csrf_token()
    except Exception:
        context.csrf_token = ''
