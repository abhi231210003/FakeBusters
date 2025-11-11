from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, make_response
import pickle
import numpy as np
import pandas as pd
import io
import os
import uuid

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
    # CSV preview (first-5 rows) — show once
    csv_preview = session.pop('csv_preview', None)
    # Keep download path/name in session until user clears or downloads
    csv_download_name = session.get('csv_download_name')
    csv_available = bool(session.get('csv_path'))

    # allow explicit tab override via query param (e.g. after CSV upload)
    qtab = request.args.get('tab', '').lower() if request.args else ''
    if qtab in ('csv', 'single'):
        active_tab = qtab
    else:
        # default: CSV ONLY if there's a fresh csv_preview (just uploaded), otherwise single
        # Don't stay on CSV tab just because a file is available from an old upload
        active_tab = 'csv' if csv_preview else 'single'

    return render_template("index.html", result=result, task=task, message=message, probability=probability, explanation=explanation, csv_preview=csv_preview, csv_download_name=csv_download_name, csv_available=csv_available, active_tab=active_tab)

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


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Accept a CSV upload, run batch predictions and return a CSV with added columns.

    Expected form fields:
    - file: uploaded csv file
    - task: 'fake_news' or 'spam'
    - text_column: optional, the name of the column containing text (try common defaults if not provided)
    - threshold: optional numeric percentage (0-100) to override model threshold for positive label
    """
    if 'file' not in request.files:
        return redirect(url_for('home'))

    uploaded = request.files['file']
    if uploaded.filename == '':
        return redirect(url_for('home'))

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        session['result'] = f"❌ Failed to read CSV: {e}"
        return redirect(url_for('home'))

    task = request.form.get('task', 'fake_news')
    text_col = request.form.get('text_column', '').strip()
    threshold = request.form.get('threshold', '').strip()
    thr = None
    try:
        if threshold != '':
            thr = float(threshold) / 100.0
    except Exception:
        thr = None

    # Try to find a sensible text column if not specified
    if not text_col or text_col not in df.columns:
        candidates = [c for c in ['message', 'text', 'content', 'headline', 'body'] if c in df.columns]
        if candidates:
            text_col = candidates[0]
        else:
            # fallback to first string-like column
            for c in df.columns:
                if pd.api.types.is_string_dtype(df[c]):
                    text_col = c
                    break

    if not text_col or text_col not in df.columns:
        session['result'] = '❌ Could not determine text column in CSV. Provide `text_column` or include a column named message/text/content.'
        return redirect(url_for('home'))

    texts = df[text_col].fillna('').astype(str).tolist()

    # choose model and vectorizer
    if task == 'spam':
        model = spam_model
        vectorizer = spam_vectorizer
    else:
        model = fake_news_model
        vectorizer = fake_news_vectorizer

    try:
        X = vectorizer.transform(texts)
    except Exception as e:
        session['result'] = f'❌ Vectorization failed: {e}'
        return redirect(url_for('home'))

    try:
        preds = model.predict(X)
    except Exception as e:
        session['result'] = f'❌ Prediction failed: {e}'
        return redirect(url_for('home'))

    # probabilities if available
    prob_list = [None] * len(texts)
    if hasattr(model, 'predict_proba'):
        try:
            proba_all = model.predict_proba(X)
            classes = list(getattr(model, 'classes_', []))
            for i, proba in enumerate(proba_all):
                if task == 'spam':
                    # probability of Spam (label 1) if available
                    if 1 in classes:
                        prob = proba[classes.index(1)]
                    elif 0 in classes:
                        prob = 1.0 - proba[classes.index(0)]
                    else:
                        prob = proba.max()
                else:
                    # fake_news: probability of Fake (label 0)
                    if 0 in classes:
                        prob = proba[classes.index(0)]
                    elif 1 in classes:
                        prob = 1.0 - proba[classes.index(1)]
                    else:
                        prob = proba.max()
                prob_list[i] = round(float(prob) * 100.0, 1)
        except Exception:
            prob_list = [None] * len(texts)

    # assemble human-friendly labels, optionally using threshold
    label_list = []
    for i, p in enumerate(preds):
        prob_pct = prob_list[i]
        if thr is not None and prob_pct is not None:
            # threshold drives decision
            use_fake = (prob_pct / 100.0) >= thr
            if task == 'spam':
                label = '📩 Spam Message' if use_fake else '✉️ Not Spam'
            else:
                label = '🚨 Fake News' if use_fake else '✅ Real News'
        else:
            # fallback to model prediction mapping used elsewhere
            if task == 'spam':
                label = '✉️ Not Spam' if int(p) == 0 else '📩 Spam Message'
            else:
                label = '✅ Real News' if int(p) == 1 else '🚨 Fake News'
        label_list.append(label)

    # attach results to DataFrame
    df['prediction'] = label_list
    df['probability'] = prob_list

    # derive a safe filename for download and save to server-side temp folder
    original_name = getattr(uploaded, 'filename', 'upload')
    if original_name.lower().endswith('.csv'):
        out_name = original_name[:-4] + '_predictions.csv'
    else:
        out_name = original_name + '_predictions.csv'

    # ensure predictions folder exists
    preds_dir = os.path.join(os.path.dirname(__file__), 'predictions')
    os.makedirs(preds_dir, exist_ok=True)

    unique_id = uuid.uuid4().hex
    safe_filename = f"{unique_id}_{out_name}"
    save_path = os.path.join(preds_dir, safe_filename)
    try:
        df.to_csv(save_path, index=False)
    except Exception as e:
        session['result'] = f'❌ Failed to save predictions file: {e}'
        return redirect(url_for('home'))

    # store a small preview (first 5 rows) in session for display
    preview_df = df.head(5).copy()
    def _safe_val(v):
        try:
            if pd.isna(v):
                return ''
        except Exception:
            pass
        s = str(v)
        if len(s) > 200:
            return s[:197] + '...'
        return s

    preview_records = []
    for _, row in preview_df.iterrows():
        rec = {}
        for col in preview_df.columns:
            rec[col] = _safe_val(row[col])
        preview_records.append(rec)
    session['csv_preview'] = preview_records
    session['csv_download_name'] = out_name
    session['csv_path'] = save_path

    # redirect to home where preview will be shown (PRG)
    session['result'] = f'✅ CSV processed: {len(df)} rows. Preview below.'
    # include a query param to force the CSV tab on the redirected page
    return redirect(url_for('home', tab='csv'))


@app.route('/api/predict_text', methods=['POST'])
def api_predict_text():
    """Simple JSON API to predict a single text.

    Expects JSON: {"task": "fake_news"|"spam", "message": "..."}
    Returns JSON: {"prediction": "...", "probability": 12.3, "explanation": "..."}
    """
    payload = request.get_json(silent=True)
    if not payload:
        return make_response(jsonify({'error': 'Invalid or missing JSON payload'}), 400)

    user_input = payload.get('message', '')
    selected_task = payload.get('task', 'fake_news')
    if not user_input or not user_input.strip():
        return make_response(jsonify({'error': 'Empty message provided'}), 400)

    try:
        if selected_task == 'spam':
            input_vector = spam_vectorizer.transform([user_input])
            prediction = spam_model.predict(input_vector)[0]
            result = "✉️ Not Spam" if int(prediction) == 0 else "📩 Spam Message"
        else:
            input_vector = fake_news_vectorizer.transform([user_input])
            prediction = fake_news_model.predict(input_vector)[0]
            result = "✅ Real News" if int(prediction) == 1 else "🚨 Fake News"
    except Exception as e:
        return make_response(jsonify({'error': f'Prediction failed: {e}'}), 500)

    probability = None
    try:
        if selected_task == 'fake_news' and hasattr(fake_news_model, 'predict_proba'):
            proba = fake_news_model.predict_proba(input_vector)[0]
            classes = list(getattr(fake_news_model, 'classes_', []))
            if 0 in classes:
                probability = proba[classes.index(0)]
            elif 1 in classes:
                probability = 1.0 - proba[classes.index(1)]
            else:
                probability = proba.max()
        elif selected_task == 'spam' and hasattr(spam_model, 'predict_proba'):
            proba = spam_model.predict_proba(input_vector)[0]
            classes = list(getattr(spam_model, 'classes_', []))
            if 1 in classes:
                probability = proba[classes.index(1)]
            elif 0 in classes:
                probability = 1.0 - proba[classes.index(0)]
            else:
                probability = proba.max()
    except Exception:
        probability = None

    prob_pct = None
    if probability is not None:
        try:
            prob_pct = round(float(probability) * 100.0, 1)
        except Exception:
            prob_pct = None

    # short keyword-based explanation fallback
    explanation = None
    text_lower = user_input.lower()
    if selected_task == 'fake_news':
        keywords_fake = ['shocking', 'unbelievable', 'secret', "won't believe", 'breaking', 'exclusive', 'conspiracy', 'miracle']
        found = [k for k in keywords_fake if k in text_lower]
        if found:
            explanation = f"Contains sensational words like {', '.join(found[:3])}."
        else:
            explanation = "Prediction based on textual features and patterns."
    else:
        keywords_spam = ['click', 'win', 'prize', 'congratulations', 'free', 'urgent', 'claim', 'offer', 'buy now']
        found = [k for k in keywords_spam if k in text_lower]
        if found:
            explanation = f"Contains suspicious phrases like {', '.join(found[:3])}."
        else:
            explanation = "Prediction based on textual features and patterns."

    return jsonify({
        'prediction': result,
        'probability': prob_pct,
        'explanation': explanation
    })


@app.route('/download_predictions')
def download_predictions():
    # send latest predictions file saved in session
    save_path = session.get('csv_path')
    download_name = session.get('csv_download_name') or 'predictions.csv'
    if not save_path or not os.path.exists(save_path):
        session['result'] = '❌ No predictions file available for download.'
        return redirect(url_for('home'))
    return send_file(save_path, as_attachment=True, download_name=download_name)


@app.route('/clear_predictions')
def clear_predictions():
    # remove preview and file reference from session and delete file
    save_path = session.pop('csv_path', None)
    session.pop('csv_download_name', None)
    session.pop('csv_preview', None)
    if save_path and os.path.exists(save_path):
        try:
            os.remove(save_path)
        except Exception:
            pass
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)
