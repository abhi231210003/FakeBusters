from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# ====== LOAD MODELS AND VECTORIZERS ======
fake_news_model = pickle.load(open("fake_news_model.pkl", "rb"))
fake_news_vectorizer = pickle.load(open("fake_news_vectorizer.pkl", "rb"))
spam_model = pickle.load(open("spam_model.pkl", "rb"))
spam_vectorizer = pickle.load(open("spam_vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     user_input = request.form["message"]
#     selected_task = request.form["task"]

#     if not user_input.strip():
#         return render_template("index.html", result="⚠️ Please enter some text!", task=selected_task)

#     # ====== Fake News Detection ======
#     if selected_task == "fake_news":
#         input_vector = fake_news_vectorizer.transform([user_input])
#         prediction = fake_news_model.predict(input_vector)[0]
#         result = "✅ Real News" if prediction == 1 else "🚨 Fake News"

#     # ====== Spam Email Detection ======
#     elif selected_task == "spam":
#         input_vector = spam_vectorizer.transform([user_input])
#         prediction = spam_model.predict(input_vector)[0]
#         result = "✉️ Not Spam" if prediction == 0 else "📩 Spam Message"

#     else:
#         result = "Invalid Task Selected."

#     return render_template("index.html", result=result, task=selected_task, message=user_input)

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["message"]
    selected_task = request.form["task"]

    if not user_input.strip():
        return render_template("index.html", result="⚠️ Please enter some text!", task=selected_task)

    try:
        # ====== Fake News Detection ======
        if selected_task == "fake_news":
            input_vector = fake_news_vectorizer.transform([user_input])
            prediction = fake_news_model.predict(input_vector)[0]
            result = "✅ Real News" if prediction == 1 else "🚨 Fake News"

        # ====== Spam Email Detection ======
        elif selected_task == "spam":
            input_vector = spam_vectorizer.transform([user_input])
            prediction = spam_model.predict(input_vector)[0]
            result = "✉️ Not Spam" if prediction == 0 else "📩 Spam Message"

        else:
            result = "Invalid Task Selected."

    except ValueError as e:
        result = f"❌ Feature mismatch error: {str(e)}. Please ensure model and vectorizer are from same training session."

    return render_template("index.html", result=result, task=selected_task, message=user_input)



if __name__ == "__main__":
    app.run(debug=True)
