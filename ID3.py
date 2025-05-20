import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Mild', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Low', 'Low'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

X = pd.get_dummies(df[['Weather', 'Temperature', 'Humidity']])
y = df['Play'].map({'No': 0, 'Yes': 1})

#Train Decision Tree with ID3
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

print("Decision Tree Rules:\n")
print(export_text(model, feature_names=list(X.columns)))

# 5. Predict a new sample: Weather=Rain, Temperature=Mild, Humidity=Low
sample = pd.DataFrame([{
    'Weather_Overcast': 0, 'Weather_Rain': 1, 'Weather_Sunny': 0,
    'Temperature_Hot': 0, 'Temperature_Mild': 1, 'Temperature_Low': 0,
    'Humidity_High': 0, 'Humidity_Low': 1
}])

sample = sample[X.columns]

prediction = model.predict(sample)[0]
print("\nPrediction for [Rain, Mild, Low]:", "Play=Yes" if prediction else "Play=No")
