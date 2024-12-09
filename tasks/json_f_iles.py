import json

blocked_reasons = [
    "Violation of terms of service",
    "Suspicious activity detected",
    "Non-payment of subscription fees",
    "Spamming or abusive behavior",
    "Inappropriate content posted",
    "Multiple failed login attempts",
    "Unauthorized access attempts",
    "Account reported by multiple users",
    "Breaching confidentiality agreements",
    "Use of prohibited software or bots",
    "Misrepresentation of identity",
    "Non-compliance with community guidelines",
    "Fraudulent transactions detected",
    "Harassment of other users",
    "Security policy violation"
]

# Save the list to a JSON file
with open('blocked_reasons.json', 'w') as json_file:
    json.dump(blocked_reasons, json_file)
