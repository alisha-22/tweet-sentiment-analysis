"""Live demo: tweet sentiment analysis with Logistic Regression and Naive Bayes.

The models are trained in the notebook and saved with pickle (lr_model.pkl, nb_model.pkl).
This app only loads them, so no training happens here.
"""
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import sklearn
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay

from utils import process_tweet

st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="wide")

HERE = Path(__file__).parent
MODEL_FILES = {"lr": HERE / "lr_model.pkl", "nb": HERE / "nb_model.pkl"}


# --------------------------------------------------------------------------- load saved models (cached)
@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    nltk.download("stopwords", quiet=True)           # needed by process_tweet
    with open(MODEL_FILES["lr"], "rb") as f:
        lr = pickle.load(f)
    with open(MODEL_FILES["nb"], "rb") as f:
        nb = pickle.load(f)
    # the vectorizer is rebuilt from the saved vocabulary (it only maps words to columns)
    vectorizer = CountVectorizer(analyzer=lambda tokens: tokens, vocabulary=nb["vocabulary"])
    return lr, nb, vectorizer


missing = [p.name for p in MODEL_FILES.values() if not p.exists()]
if missing:
    st.error(f"Model file(s) not found: {', '.join(missing)}. Run the notebook first (Part D) to create them, "
             "and keep them in the same folder as this app.")
    st.stop()

LR, NB, VECTORIZER = load_models()
INFO = LR["info"]
if INFO.get("sklearn_version") != sklearn.__version__:
    st.warning(f"The models were saved with scikit-learn {INFO.get('sklearn_version')} but this app runs "
               f"{sklearn.__version__}. If you see errors, install the same version.")


def predict_lr(text):
    tokens = process_tweet(text)
    feats = pd.DataFrame({"pos_freq": [sum(LR["pos_map"].get(w, 0) for w in tokens)],
                          "neg_freq": [sum(LR["neg_map"].get(w, 0) for w in tokens)]})
    return float(LR["model"].predict_proba(feats)[0, 1]), feats.iloc[0]


def predict_nb(text):
    tokens = process_tweet(text)
    log_proba = NB["classifier"].predict_log_proba(VECTORIZER.transform([tokens]))[0]
    score = float(log_proba[1] - log_proba[0])        # log-odds: above 0 = positive
    table = NB["word_table"]
    known = [w for w in tokens if w in table.index]
    contrib = pd.DataFrame({"word": known, "log likelihood": [table.loc[w, "loglikelihood"] for w in known]})
    return score, tokens, contrib


# --------------------------------------------------------------------------- header / sidebar
st.title("💬 Tweet Sentiment Analyzer")
st.markdown("Type a tweet and see how **Logistic Regression** and **Naive Bayes** classify it. "
            f"Both models were trained in a notebook on {INFO['n_train']:,} labeled tweets using NLTK, pandas "
            "and scikit-learn, then saved with pickle.")

with st.sidebar:
    st.header("About the data")
    st.metric("Training tweets", f"{INFO['n_train']:,}")
    st.metric("Test tweets", f"{INFO['n_test']:,}")
    st.metric("Vocabulary (stemmed words)", f"{INFO['vocab_size']:,}")
    st.caption("Source: NLTK `twitter_samples` (5,000 positive + 5,000 negative tweets).")

tab_try, tab_results, tab_words, tab_about = st.tabs(["Try it", "Model results", "Word explorer", "How it works"])

# --------------------------------------------------------------------------- tab 1: try it
with tab_try:
    examples = {
        "Happy": "I am happy because I am learning :)",
        "Sad": "you are bad :(",
        "Neutral-ish": "She smiled.",
        "Mixed": "This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!",
    }
    if "tweet" not in st.session_state:
        st.session_state.tweet = examples["Happy"]

    def set_example(text):
        st.session_state.tweet = text

    st.caption("Try an example:")
    cols = st.columns(len(examples))
    for col, (name, text) in zip(cols, examples.items()):
        col.button(name, on_click=set_example, args=(text,), width="stretch")

    tweet = st.text_area("Your tweet", key="tweet", height=100)

    if tweet.strip():
        lr_prob, feats = predict_lr(tweet)
        nb_score, tokens, contrib = predict_nb(tweet)

        if not tokens:
            st.warning("No meaningful words were left after cleaning (stopwords and punctuation are removed). "
                       "Try a longer tweet.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Logistic Regression")
            st.metric("Prediction", "Positive 😊" if lr_prob > 0.5 else "Negative 😞",
                      f"{lr_prob * 100:.1f}% chance positive", delta_color="off")
            st.progress(lr_prob)
            st.caption(f"Features: positive-word score **{int(feats['pos_freq']):,}**, "
                       f"negative-word score **{int(feats['neg_freq']):,}**")
        with c2:
            st.subheader("Naive Bayes")
            st.metric("Prediction", "Positive 😊" if nb_score > 0 else "Negative 😞",
                      f"score {nb_score:+.2f} (above 0 = positive)", delta_color="off")
            st.progress(float(1 / (1 + math.exp(-nb_score))) if abs(nb_score) < 700 else float(nb_score > 0))
            st.caption(f"Log prior {NB['logprior']:+.2f} + sum of word log likelihoods "
                       f"{contrib['log likelihood'].sum():+.2f}")

        st.divider()
        st.subheader("What the models saw")
        st.write("Cleaned tokens:", " ".join(f"`{t}`" for t in tokens) if tokens else "_none_")
        if len(contrib):
            st.caption("Each word's contribution to the Naive Bayes score. Positive values push towards positive, "
                       "negative values towards negative. Words not seen in training are ignored.")
            chart = contrib.groupby("word", sort=False)["log likelihood"].sum()
            st.bar_chart(chart, horizontal=True)

# --------------------------------------------------------------------------- tab 2: results
with tab_results:
    st.subheader(f"Performance on {INFO['n_test']:,} unseen test tweets")
    models = {"Logistic Regression": LR, "Naive Bayes": NB}
    metrics = pd.DataFrame({
        name: {"Accuracy": m["scores"]["accuracy"], "Precision": m["scores"]["precision"],
               "Recall": m["scores"]["recall"], "F1": m["scores"]["f1"]}
        for name, m in models.items()}).T
    st.dataframe(metrics.style.format("{:.2%}"), width="stretch")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, m) in zip(axes, models.items()):
        ConfusionMatrixDisplay(m["confusion_matrix"], display_labels=["Negative", "Positive"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Misclassified tweets")
    model_name = st.radio("Model", list(models), horizontal=True)
    wrong = models[model_name]["misclassified"]
    st.caption(f"{len(wrong)} of {INFO['n_test']:,} tweets were misclassified.")
    st.dataframe(wrong, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- tab 3: word explorer
with tab_words:
    st.subheader("Which words push a tweet positive or negative?")
    min_count = st.slider("Minimum total count in training tweets", 5, 200, 30)
    top_n = st.slider("Words to show", 5, 25, 10)
    freq = NB["word_table"].assign(total=lambda d: d["pos"] + d["neg"])
    freq = freq[freq["total"] >= min_count]
    show = ["pos", "neg", "ratio", "loglikelihood"]
    a, b = st.columns(2)
    with a:
        st.markdown("**Most positive**")
        st.dataframe(freq.sort_values("loglikelihood", ascending=False).head(top_n)[show]
                     .rename(columns={"loglikelihood": "log likelihood"}), width="stretch")
    with b:
        st.markdown("**Most negative**")
        st.dataframe(freq.sort_values("loglikelihood").head(top_n)[show]
                     .rename(columns={"loglikelihood": "log likelihood"}), width="stretch")
    st.caption("`pos` / `neg` = how often the (stemmed) word appears in positive / negative training tweets. "
               "`ratio` = (pos + 1) / (neg + 1).")

    st.divider()
    query = st.text_input("Look up a word (it is stemmed automatically, e.g. 'happy' becomes 'happi')")
    if query.strip():
        stems = process_tweet(query)
        if stems and stems[0] in NB["word_table"].index:
            st.dataframe(NB["word_table"].loc[[stems[0]], show].rename(columns={"loglikelihood": "log likelihood"}),
                         width="stretch")
        else:
            st.info("That word was not seen in the training tweets (or it is a stopword).")

# --------------------------------------------------------------------------- tab 4: how it works
with tab_about:
    st.subheader("The pipeline")
    st.markdown("""
1. **Load** 5,000 positive and 5,000 negative tweets into a pandas DataFrame.
2. **Split** 80% train / 20% test with scikit-learn (stratified).
3. **Clean** each tweet with NLTK: remove handles, links, stopwords and punctuation, then tokenize and stem.
4. **Count** how often each word appears in each class (training set only).
5. **Logistic Regression:** every tweet becomes two numbers, a positive-word score and a negative-word score.
6. **Naive Bayes:** word counts with Laplace smoothing (add 1), a log prior and per-word log likelihoods.
7. **Evaluate** with accuracy, precision, recall, F1 and confusion matrices.
8. **Save** both trained models with pickle (`lr_model.pkl`, `nb_model.pkl`). This app loads them, so it never retrains.
""")
    st.info("Both models rely heavily on emoticons like :) and :( in this dataset, so tweets without them are "
            "a tougher test. Try removing the emoticon from an example above and watch the scores change.")
