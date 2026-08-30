import requests
API_KEY = "e5e0c03a2b47de2811e46326531a7565"
CITY = "Karachi"
COUNTRY = "PK"
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"
response = requests.get(URL)
weather_data = response.json()
print(weather_data)