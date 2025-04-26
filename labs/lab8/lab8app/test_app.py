import requests

# Define the URL of the local FastAPI server
url = "http://127.0.0.1:8000/predict/"

# Example input data to send for prediction
data = {
    "Income": 100.0,
    "Limit": 5000.0,
    "Cards": 3.0,
    "Age": 45.0,
    "Education": 16.0,
    "Gender": 1.0,
    "Student": 0.0,
    "Married": 1.0,
    "Ethnicity": 2.0,
    "Balance": 400.0
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response from the FastAPI app
print(response.json())
