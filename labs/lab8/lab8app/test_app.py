import requests

# Define the URL of the local FastAPI server
url = "http://127.0.0.1:8000/predict/"

# Example input data to send for prediction
data = {
    "Gender": 0,
    "Student": 1,
    "Married": 1,
    "Ethnicity": 1,
    "Income": 50000,
    "Limit": 15000,
    "Cards": 3,
    "Age": 25,
    "Education": 3,
    "Balance": 2000
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response from the FastAPI app
print(response.json())
