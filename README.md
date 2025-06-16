# Digital-Blinc

# 🧠 Emoji Predictor using Machine Learning

The **Emoji Predictor** is an AI/ML-based project that interprets text input and returns an emoji that best represents the emotional or contextual meaning of the phrase. Designed using traditional machine learning techniques, this solution is lightweight, interpretable, and ideal for educational purposes, internships, or AI mini-projects.

## 📌 Project Overview

Text-based communication often lacks emotional depth. Emojis bridge this gap by visually expressing emotions, tone, or intent. This project builds a simple yet effective model that predicts the appropriate emoji for a given phrase or sentence using a custom dataset and ML pipeline (TF-IDF + Logistic Regression).

## 🚀 Key Features

- ✅ Predicts emoji from free-form text input
- ✅ Uses a curated training set for mood-to-emoji mapping
- ✅ Utilizes `TfidfVectorizer` and `LogisticRegression` from scikit-learn
- ✅ Easy to train, modify, and extend
- ✅ Saves model for future inference using `joblib`
- ✅ No deep learning or GPU required — runs on standard machines

## 🧠 Technology Stack

| Component         | Technology        |
|------------------|-------------------|
| Programming Lang | Python 3.x         |
| Libraries        | `pandas`, `scikit-learn`, `joblib` |
| ML Model         | Logistic Regression |
| Feature Extraction | TF-IDF Vectorizer  |
| Output Format    | Emoji (Unicode)     |

## 📁 Project Structure
emoji_predictor_project/
│
├── emoji_predictor_ml.py # Main script with training + prediction
├── emoji_predictor_model.pkl # Saved trained model
├── requirements.txt # List of dependencies
└── README.md # Project documentation

### Step 1: Install Dependencies
pip install pandas scikit-learn joblib

### Step 2: Run the Model
python emoji_predictor_ml.py

### Step 3: Example Usage
Type a phrase: I am feeling great today!
Predicted Emoji: 😊

🧬 How It Works

A sample dataset is created with text phrases mapped to emoji labels.
TfidfVectorizer converts text into numerical vectors.
LogisticRegression is trained on these vectors to learn the mappings.
The model is saved to disk for reuse.
On user input, the model predicts the label and returns the associated emoji.

🔢 Emoji Mapping (Partial View)

Label	Emoji	Meaning
0	😜	Playful / Silly
1	📸	Photo / Selfie
2	😍	Love / Admiration
3	😂	Funny / Hilarious
4	😉	Flirt / Witty
5	😊	Happy / Content
6	😎	Confident / Stylish
7	💯	True / Impressive
8	😢	Sad / Emotional
9	🔥	Cool / Awesome
10	🎉	Celebration / Event
11	🤔	Thinking / Curious
12	🙏	Thankful / Request
13	🥳	Party / Joyful
14	🤯	Shock / Mind-blown
15	🥺	Pleading / Emotional

📈 Future Enhancements

🔍 Use advanced models like BERT or LSTM for contextual understanding
🧑‍💻 Build a web-based interface using Streamlit or Flask
📦 Integrate with external emoji APIs (e.g., Emojinet, Twitter dataset)
🎨 Support multiple emojis for a single sentence (multi-label prediction)
🌐 Add support for different languages

📃 License

This project is released for educational and non-commercial purposes only.


