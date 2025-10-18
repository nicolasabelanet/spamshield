from spamshield.client.client import SpamShieldAPIClient

message = "WINNER!! As a valued network customer you have been selected to receive a prize reward!"
client = SpamShieldAPIClient("http://localhost:8000", api_key="dev-key")
response = client.predict([message])

print(response)
