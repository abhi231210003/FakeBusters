from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session

# ====== LOAD MODELS AND VECTORIZERS ======
fake_news_model = pickle.load(open("fake_news_model.pkl", "rb"))
fake_news_vectorizer = pickle.load(open("fake_news_vectorizer.pkl", "rb"))
spam_model = pickle.load(open("spam_model.pkl", "rb"))
spam_vectorizer = pickle.load(open("spam_vectorizer.pkl", "rb"))


@app.route("/")
def home():
    # Get result from session if it exists, then clear it
    result = session.pop('result', None)
    task = session.pop('task', None)
    message = session.pop('message', None)
    probability = session.pop('probability', None)
    explanation = session.pop('explanation', None)
    
    return render_template("index.html", result=result, task=task, message=message, probability=probability, explanation=explanation)

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["message"]
    selected_task = request.form["task"]

    if not user_input.strip():
        session['result'] = "⚠️ Please enter some text!"
        session['task'] = selected_task
        session['message'] = ""
        return redirect(url_for('home'))

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

    # compute probability if model supports it
    probability = None
    try:
        if selected_task == "fake_news" and hasattr(fake_news_model, 'predict_proba'):
            proba = fake_news_model.predict_proba(input_vector)[0]
            classes = list(getattr(fake_news_model, 'classes_', []))
            # probability of Fake News (label 0) if exists, otherwise try fallback
            if 0 in classes:
                idx = classes.index(0)
                probability = proba[idx]
            elif 1 in classes:
                probability = 1.0 - proba[classes.index(1)]
            else:
                # fallback: take probability of predicted class
                probability = proba.max()
        elif selected_task == "spam" and hasattr(spam_model, 'predict_proba'):
            proba = spam_model.predict_proba(input_vector)[0]
            classes = list(getattr(spam_model, 'classes_', []))
            # probability of Spam (label 1) if exists
            if 1 in classes:
                idx = classes.index(1)
                probability = proba[idx]
            elif 0 in classes:
                probability = 1.0 - proba[classes.index(0)]
            else:
                probability = proba.max()
    except Exception:
        probability = None

    # format probability as percentage if available
    prob_pct = None
    if probability is not None:
        try:
            prob_pct = round(float(probability) * 100, 1)
        except Exception:
            prob_pct = None

    # Store results in session and redirect to home
    session['result'] = result
    session['task'] = selected_task
    session['message'] = user_input
    session['probability'] = prob_pct
    # compute short explanation/reason for prediction
    explanation = None
    try:
        # Try feature-based explanation if model exposes coefficients and vectorizer feature names
        if selected_task == 'fake_news' and hasattr(fake_news_model, 'coef_') and hasattr(fake_news_vectorizer, 'get_feature_names_out'):
            features = fake_news_vectorizer.get_feature_names_out()
            coef = np.array(fake_news_model.coef_)
            vec = input_vector.toarray()[0]
            # choose coef row (binary classifiers often have shape (1, n_features))
            if coef.ndim == 2 and coef.shape[0] == 1:
                coef_row = coef[0]
            elif coef.ndim == 2:
                # try to find row corresponding to predicted class
                classes = list(getattr(fake_news_model, 'classes_', []))
                if prediction in classes:
                    coef_row = coef[classes.index(prediction)]
                else:
                    coef_row = coef[0]
            else:
                coef_row = coef
            contrib = coef_row * vec
            top_idx = np.argsort(contrib)[-5:][::-1]
            top_words = [features[i] for i in top_idx if vec[i] > 0]
            if top_words:
                explanation = f"Signals: {', '.join(top_words)}"
        elif selected_task == 'spam' and hasattr(spam_model, 'coef_') and hasattr(spam_vectorizer, 'get_feature_names_out'):
            features = spam_vectorizer.get_feature_names_out()
            coef = np.array(spam_model.coef_)
            vec = input_vector.toarray()[0]
            if coef.ndim == 2 and coef.shape[0] == 1:
                coef_row = coef[0]
            elif coef.ndim == 2:
                classes = list(getattr(spam_model, 'classes_', []))
                if prediction in classes:
                    coef_row = coef[classes.index(prediction)]
                else:
                    coef_row = coef[0]
            else:
                coef_row = coef
            contrib = coef_row * vec
            top_idx = np.argsort(contrib)[-5:][::-1]
            top_words = [features[i] for i in top_idx if vec[i] > 0]
            if top_words:
                explanation = f"Signals: {', '.join(top_words)}"
    except Exception:
        explanation = None

    # Fallback keyword-based explanations if feature-based failed or empty
    if not explanation:
        text_lower = user_input.lower()
        if selected_task == 'fake_news':
            keywords_fake = ['shocking', 'unbelievable', 'secret', 'won\'t believe', 'breaking', 'exclusive', 'conspiracy', 'miracle']
            found = [k for k in keywords_fake if k in text_lower]
            if found:
                explanation = f"Contains sensational words like {', '.join(found[:3])}, which are common in misleading articles."
            else:
                explanation = "Predicted based on textual patterns and phrasing that resemble misleading or low-quality sources."
        elif selected_task == 'spam':
            keywords_spam = ['click', 'win', 'prize', 'congratulations', 'free', 'urgent', 'claim', 'offer', 'buy now']
            found = [k for k in keywords_spam if k in text_lower]
            if found:
                explanation = f"Contains suspicious phrases like {', '.join(found[:3])} often seen in spam."
            else:
                explanation = "Predicted because the message structure and vocabulary match common spam patterns."

    session['explanation'] = explanation
    
    return redirect(url_for('home'))



if __name__ == "__main__":
    app.run(debug=True)
