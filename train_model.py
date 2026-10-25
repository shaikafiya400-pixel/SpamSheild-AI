
import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent

URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

print("Downloading the UCI spam dataset...")

with urllib.request.urlopen(URL, timeout=60) as response:
    archive_data = response.read()

with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
    with archive.open("SMSSpamCollection") as file:
        df = pd.read_csv(
            file,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8"
        )

print("Dataset loaded:", len(df), "messages")

# Additional email examples to improve email-related coverage.
email_examples = [
    ("spam", "URGENT: Your account has been suspended. Verify now."),
    ("spam", "Click here to claim your free cash prize today."),
    ("spam", "Congratulations, you have won a lottery. Claim your reward."),
    ("spam", "Your bank account is locked. Verify your password immediately."),
    ("spam", "You have been selected for a free gift. Click to claim."),
    ("spam", "Limited time offer. Earn money fast with no investment."),
    ("spam", "Your package is on hold. Pay a fee using this link."),
    ("spam", "Confirm your account details immediately to avoid suspension."),
    ("spam", "You won a free mobile phone. Send your personal details."),
    ("spam", "Act now to receive your guaranteed cash reward."),
    ("spam", "Your mailbox is full. Click here to upgrade your account."),
    ("spam", "We detected unusual activity. Enter your password to continue."),
    ("ham", "Hi, are we still meeting at the office today?"),
    ("ham", "Please find the meeting notes attached."),
    ("ham", "Your appointment is confirmed for tomorrow at ten."),
    ("ham", "Could you review the report and send me your feedback?"),
    ("ham", "The college meeting has been moved to Friday."),
    ("ham", "Thank you for contacting our support team."),
    ("ham", "Your order confirmation and receipt are attached."),
    ("ham", "Please reset your password using the official company website."),
    ("ham", "The monthly statement is available in your bank application."),
    ("ham", "Hello, I will call you after the meeting."),
    ("ham", "Your interview is scheduled for Monday morning."),
    ("ham", "Please reply to this email if you need further assistance.")
]

extra_df = pd.DataFrame(email_examples, columns=["label", "text"])
df = pd.concat([df, extra_df], ignore_index=True)
df = df.dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", MultinomialNB(alpha=0.3))
])

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nEvaluation report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, BASE_DIR / "model.joblib")
print("\nModel saved successfully:", BASE_DIR / "model.joblib")

tests = [
    "URGENT: Your account has been suspended. Verify now!",
    "Congratulations, you won a free cash prize. Claim now!",
    "Hi, are we still meeting at the office today?"
]

print("\nSample predictions:")
for message in tests:
    prediction = model.predict([message])[0]
    spam_index = list(model.classes_).index("spam")
    probability = model.predict_proba([message])[0][spam_index]

    print(f"\nMessage: {message}")
    print(f"Prediction: {prediction.upper()}")
    print(f"Spam probability: {probability:.1%}")
