import requests

data = {
    "ID": 2,
    "LicAge": 364,
    "Gender": "Female",
    "MariStat": "Other",
    "SocioCateg": "CSP55",
    "VehUsage": "Private+trip to office",
    "DrivAge": 52,
    "HasKmLimit": 0,
    "BonusMalus": 50,
    "OutUseNb": 0,
    "RiskArea": 8
    }


def send_json(data):
    url = 'http://127.0.0.1:5000/predict'
    headers = {'content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response


if __name__ == '__main__':
    response = send_json(data)
    print(response.json())