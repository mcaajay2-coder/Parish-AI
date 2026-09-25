
def resolve_user_parish(user_email):
    if not user_email or user_email in ['Administrator', 'admin@example.com']:
        return None
        
    # 1. Direct match on Member.email -> parish_id
    member_parish = frappe.db.get_value('Member', {'email': user_email}, 'parish_id')
    if member_parish:
        return member_parish
        
    # 2. Direct match on Parish.email -> name
    parish_by_email = frappe.db.get_value('Parish', {'email': user_email}, 'name')
    if parish_by_email:
        return parish_by_email
        
    # 3. Match Parish by parish_priest name
    try:
        user_doc = frappe.get_doc('User', user_email)
        full_name = user_doc.full_name or ''
        clean_name = full_name.replace('Fr.', '').replace('Fr', '').replace('Rev.', '').replace('SDB', '').strip()
        if clean_name and len(clean_name) > 2:
            match = frappe.db.sql("""SELECT name FROM `tabParish` WHERE parish_priest LIKE %s LIMIT 1""", (f'%{clean_name}%',), as_dict=True)
            if match:
                return match[0]['name']
    except Exception:
        pass
        
    # 4. Email prefix match on tabParish
    try:
        prefix = user_email.split('@')[0].split('_')[0].replace('priest', '').replace('parish', '').strip()
        if len(prefix) > 2:
            match = frappe.db.sql("""SELECT name FROM `tabParish` WHERE LOWER(name) LIKE %s OR LOWER(parish_name) LIKE %s LIMIT 1""", (f'%{prefix}%', f'%{prefix}%'), as_dict=True)
            if match:
                return match[0]['name']
    except Exception:
        pass
        
    return None

import frappe
import os

no_cache = 1

def get_context(context):
    context.no_sidebar = 1
    context.no_header = 1
    context.no_footer = 1
    
    if frappe.session.user == 'Guest':
        frappe.redirect('/login?redirect-to=/koinonia_chat')
        
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
        parish = resolve_user_parish(user) or 'Yelagiri Parish'
        context.badge_text = parish
        context.user_parish = parish
        context.user_diocese = 'Vellore Diocese'
    else:
        context.user_role = 'Parishioner'
        parish = resolve_user_parish(user) or 'Yelagiri Parish'
        context.badge_text = parish
        context.user_parish = parish
        context.user_diocese = 'Vellore Diocese'

    try:
        context.csrf_token = frappe.sessions.get_csrf_token()
    except Exception:
        context.csrf_token = ''
