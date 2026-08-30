import joblib
import pandas as pd

model = joblib.load("models/aqi_model.pkl")

new_weather = pd.DataFrame({
    "Max Temperature":[35],
    "Min Temperature":[28],
    "Rain":[0],
    "Wind Speed":[15]
})

prediction = model.predict(new_weather)

print("Predicted AQI:", round(prediction[0],2))