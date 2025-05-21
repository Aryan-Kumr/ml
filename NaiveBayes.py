import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

data = {
    'text': [
        'Win money now', 'Limited offer just for you', 'Important meeting tomorrow',
        'Call mom today', 'Free entry in a contest', 'Earn dollars fast',
        'Lunch at 1 PM', 'Project deadline is close', 'Buy cheap meds online',
        'Let’s catch up soon'
    ],
    'label': ['spam', 'spam', 'not spam', 'not spam', 'spam', 'spam', 'not spam', 'not spam', 'spam', 'not spam']
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_emails = ['Congratulations, you have won a prize', 'Team meeting at 10 AM']

X_new = vectorizer.transform(new_emails)

predictions = model.predict(X_new)

for text, pred in zip(new_emails, predictions):
    print(f"Text: '{text}' → Prediction: {pred}")
