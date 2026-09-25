app_name = "koinonia_assistant"
app_title = "Koinonia Assistant"
app_publisher = "Google Deepmind"
app_description = "Sacrament and Diocesan Data Assistant"
app_email = "ajay@gmail.com"
app_license = "mit"

# Permissions
# -----------
# Permissions evaluated in scripted ways

permission_query_conditions = {
	"Member": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Family": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Baptism": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Communion": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Confirmation": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Marriage": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Death": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Anointing Of Sick": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Parish": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Vicariate": "koinonia_assistant.permissions.get_permission_query_conditions",
	"Diocese": "koinonia_assistant.permissions.get_permission_query_conditions",
}

has_permission = {
	"Member": "koinonia_assistant.permissions.has_permission",
	"Family": "koinonia_assistant.permissions.has_permission",
	"Baptism": "koinonia_assistant.permissions.has_permission",
	"Communion": "koinonia_assistant.permissions.has_permission",
	"Confirmation": "koinonia_assistant.permissions.has_permission",
	"Marriage": "koinonia_assistant.permissions.has_permission",
	"Death": "koinonia_assistant.permissions.has_permission",
	"Anointing Of Sick": "koinonia_assistant.permissions.has_permission",
	"Parish": "koinonia_assistant.permissions.has_permission",
	"Vicariate": "koinonia_assistant.permissions.has_permission",
	"Diocese": "koinonia_assistant.permissions.has_permission",
}

# Document Events
# ---------------
# Hook on document methods and events

doc_events = {
	"DocType": {
		"on_update": "koinonia_assistant.api.sync_doctype_schema",
		"on_trash": "koinonia_assistant.api.delete_doctype_schema"
	}
}
