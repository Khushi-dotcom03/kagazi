USERS = {
    "alice": {
        "role": "engineer",
        "department": "Engineering",
        "access_level": "engineering"
    },

    "rahul": {
        "role": "finance_manager",
        "department": "Finance",
        "access_level": "finance"
    },

    "neha": {
        "role": "hr_manager",
        "department": "HR",
        "access_level": "hr"
    },

    "admin": {
        "role": "admin",
        "department": "All",
        "access_level": "admin"
    }
}

def get_user(user_id: str):
    return USERS.get(user_id)

def get_allowed_departments(user):
    if user is None:
        return []

    if user["role"] == "admin":
        return [
            "Engineering",
            "Finance",
            "HR",
            "Legal"
        ]

    return [user["department"]]