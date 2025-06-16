# emoji_predictor_ml.py
# Full Emoji Predictor using Machine Learning (scikit-learn based)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Expanded label-to-emoji mapping
label_to_emoji = {
    0: "😜", 1: "📸", 2: "😍", 3: "😂", 4: "😉",
    5: "😊", 6: "😎", 7: "💯", 8: "😢", 9: "🔥",
    10: "🎉", 11: "🤔", 12: "🙏", 13: "🥳", 14: "🤯",
    15: "🥺", 16: "😤", 17: "💔", 18: "😇", 19: "💪",
    20: "😴", 21: "👀", 22: "😅", 23: "🤗", 24: "🙄",
    25: "✨", 26: "🫶", 27: "😈", 28: "👑", 29: "🧠"
}

# Step 1: Create or load a small custom dataset
# Example mock dataset
sample_data = {
    "TEXT": [
        "I love you so much", "That was hilarious!", "I feel really sad",
        "What a cool trick", "Time to sleep", "You are the best!",
        "This makes me think deeply", "Feeling awesome", "What a party!",
        "Please help me", "You broke my heart", "This is amazing",
        "Stay strong", "He is watching", "This is awkward",
        "Let’s take a selfie", "I feel blessed", "You’re so smart",
        "This is so intense", "Let’s celebrate"
    ],
    "Label": [2, 3, 8, 6, 20, 5, 11, 7, 10, 12, 17, 9, 19, 21, 22, 1, 18, 29, 14, 13]
}

df = pd.DataFrame(sample_data)

# Step 2: ML Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Step 3: Train model
pipeline.fit(df['TEXT'], df['Label'])

# Save model
joblib.dump(pipeline, "D:\Digital Blinc Internship\emoji_predictor_model.pkl")

# Predict Function
def predict_emoji(text):
    model = joblib.load("D:\Digital Blinc Internship\emoji_predictor_model.pkl")
    label = model.predict([text])[0]
    return label_to_emoji.get(label, "🤖")

# Example usage
if __name__ == "__main__":
    test_input = input("Type a phrase: ")
    emoji = predict_emoji(test_input)
    print("Predicted Emoji:", emoji)
