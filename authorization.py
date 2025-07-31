# authorization.py

API_KEY = "e459c094dbdc72e9366e2645b8ab14f5c2039de3b0ab118f581d7b60d2044923"  # Replace with your actual secret token

def validate_token(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ")[1]
    return token == API_KEY
