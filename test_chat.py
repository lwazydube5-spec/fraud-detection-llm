"""
Test multi-turn /chat conversation with memory
"""
import requests
import json

BASE = 'http://localhost:8000'

claim = {
    "Month": "Jan", "WeekOfMonth": 3, "DayOfWeek": "Monday",
    "Make": "Honda", "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Feb",
    "WeekOfMonthClaimed": 2, "Sex": "Male", "MaritalStatus": "Single",
    "Age": 28, "Fault": "Policy Holder",
    "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
    "VehiclePrice": "more than 69000", "PolicyNumber": 99999,
    "RepNumber": 5, "Deductible": 400, "DriverRating": 1,
    "Days_Policy_Accident": "1 to 7", "Days_Policy_Claim": "8 to 15",
    "PastNumberOfClaims": "2 to 4", "AgeOfVehicle": "new",
    "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
    "WitnessPresent": "No", "AgentType": "External",
    "NumberOfSuppliments": "3 to 5", "AddressChange_Claim": "under 6 months",
    "NumberOfCars": "1 vehicle", "Year": 1994, "BasePolicy": "Collision"
}

history = []

def chat(message, include_claim=False):
    global history
    payload = {'message': message, 'history': history}
    if include_claim:
        payload['claim'] = claim

    print(f'\n{"="*58}')
    print(f'USER: {message}')
    print('='*58)

    response = requests.post(f'{BASE}/chat', json=payload)
    data     = response.json()

    print(f'AGENT: {data["response"]}')
    print(f'\n[{data["total_ms"]/1000:.1f}s]')

    history = data['history']

#
# Turn 1 — score the claim 
chat('Score this claim and tell me if it is worth investigating',
     include_claim=True)

# Turn 2 — follow up without resending claim 
chat('Draft an email requesting the police report and witness details')

# Turn 3 — another follow up
chat('What similar fraud patterns have been seen before?')