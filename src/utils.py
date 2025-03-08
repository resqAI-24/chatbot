import json


filepath = r"D:\NewData\Projects\Hackathons\Reckon6\chatbot\dataset\emergency_data.json"
def load_data(filepath):
    """Loads emergency dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Ensure compatibility if "emergency_contacts" is missing
    responses = data.get("emergency_responses", [])  # Default to empty list if missing
    contacts = data.get("emergency_contacts", {})  # Default to empty dict if missing

    return responses, contacts

def find_relevant_response(user_input, emergency_responses):
    """Finds the most relevant emergency response based on an exact or partial keyword match."""
    user_input = user_input.lower()
    exact_match = None
    partial_match = None

    for entry in emergency_responses:
        # Check for exact keyword match
        if user_input in entry["keywords"]:
            exact_match = entry
            break  # Prioritize exact match

        # Check for partial keyword match
        if any(keyword in user_input for keyword in entry["keywords"]):
            partial_match = entry  

    # Prioritize exact matches, then partial matches
    if exact_match:
        return exact_match["answer_en"]
    elif partial_match:
        return partial_match["answer_en"]

    return None  # No relevant response found

def find_emergency_contact(user_input, contacts):
    """Finds a relevant emergency contact based on keywords in input."""
    user_input = user_input.lower()
    
    # Mapping emergency types to their most relevant contacts
    emergency_mapping = {
        "fire": "fire_department",
        "police": "police",
        "ambulance": "ambulance",
        "disaster": "disaster_management",
        "mental health": "mental_health",
        "child help": "child_helpline",
        "women help": "women_helpline"
    }

    for key, service in emergency_mapping.items():
        if key in user_input:
            contact = contacts.get(service)
            if contact:
                return f"🚨 Emergency Contact: {contact['name']} - 📞 {contact['number']} ({contact['description']})"

    return None  # No relevant contact found
