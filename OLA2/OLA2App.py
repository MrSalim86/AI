import pandas as pd
import joblib

# Indlæs pipeline og model
pipeline = joblib.load("pipeline.pkl")
model = joblib.load("model.pkl")

# Brugerinput
print("Indtast oplysninger:")

age = int(input("Alder: "))
sex = input("Køn (male/female): ").lower()
bmi = float(input("BMI: "))
children = int(input("Antal børn: "))
smoker = input("Ryger? (yes/no): ").lower()
region = input("Region (northeast/northwest/southeast/southwest): ").lower()

# Lav dataframe
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

# Forudsig med pipeline og model
X_transformed = pipeline.transform(input_data)
prediction = model.predict(X_transformed)

print(f"\n✅ Forventet forsikringspris: {round(prediction[0], 2)} kr.")
