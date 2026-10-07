# Tweet Sentiment Analysis: Logistic Regression vs Naive Bayes

Can a model read the mood of a tweet? This project classifies tweets as **positive** or **negative** with two classic models, compares them on the same data, split and metrics, and ships them in a **live Streamlit app**.

Built with **NLTK** (text processing), **pandas** (data handling), **scikit-learn** (modeling and evaluation), **pickle** (saving the trained models) and **Streamlit** (the demo app).

## Live demo

Try it yourself: **[add your Streamlit app link here]**

Type any tweet and see how both models classify it, with the cleaned tokens, each word's contribution to the Naive Bayes score, the test-set metrics and confusion matrices, the misclassified tweets, and a word explorer. The app loads the saved models, so nothing is retrained when it starts.

## Results

Evaluated on 2,000 held-out tweets (20% of the data) that the models never saw during training.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 99.50% | 99.30% | 99.70% | 99.50% |
| Naive Bayes | 99.65% | 99.70% | 99.60% | 99.65% |

Confusion matrices (rows = true class, columns = predicted class):

| | Logistic Regression | | Naive Bayes | |
|---|---|---|---|---|
| | Pred. negative | Pred. positive | Pred. negative | Pred. positive |
| **True negative** | 993 | 7 | 997 | 3 |
| **True positive** | 3 | 997 | 4 | 996 |

Naive Bayes edges ahead by 3 tweets out of 2,000, so both models perform excellently here.

## Dataset

The `twitter_samples` corpus from NLTK: 5,000 positive and 5,000 negative tweets (10,000 in total), split 80% train / 20% test with a stratified split so both sets keep the 50/50 balance.

## Approach

1. **Load** the tweets into a pandas DataFrame with a `label` column (1 = positive, 0 = negative).
2. **Split** into train and test sets with `train_test_split`.
3. **Preprocess** each tweet with NLTK: remove handles, links, `#` signs, stopwords and punctuation, then tokenize and stem. Emoticons such as `:)` are kept because they carry sentiment.
4. **Count words per class** on the training set only, using NLTK's `ConditionalFreqDist`, so no test information leaks into the model.
5. **Logistic Regression:** each tweet is reduced to two features (the summed positive-class and negative-class word counts), scaled with `StandardScaler`, and fed to `LogisticRegression`.
6. **Naive Bayes:** a word-count matrix from `CountVectorizer` is fed to `MultinomialNB` (`alpha=1`, i.e. Laplace smoothing). The log prior and Laplace-smoothed log likelihoods are also computed with NLTK (`FreqDist`, `LaplaceProbDist`) and checked against scikit-learn's values.
7. **Evaluate** both models with accuracy, precision, recall, F1 and confusion matrices, then analyze the misclassified tweets and rank words by their positive/negative ratio.
8. **Save** both trained models with `pickle` (`lr_model.pkl`, `nb_model.pkl`) and verify that the reloaded models predict exactly like the originals.

### About the saved models

Each `.pkl` file is a dictionary holding the trained model plus what the app needs to use it (word counts, log prior, test results).

- **Logistic Regression:** the full scikit-learn pipeline (scaler + classifier) and the word-count maps that turn a tweet into its two features.
- **Naive Bayes:** the trained classifier and its vocabulary. The vectorizer itself is not pickled because it uses a `lambda`, which `pickle` cannot store. The vocabulary is everything it learned, so the app rebuilds it exactly.
- **Version note:** pickle files depend on the library versions that created them. `requirements.txt` pins the versions used, and the app warns if the scikit-learn version differs.
- **Safety note:** only load pickle files you created or trust, because unpickling can run code.

## Key takeaways

- Cleaning and stemming shrink the vocabulary and make word counts meaningful.
- Laplace smoothing stops a word unseen in one class from zeroing out a prediction.
- Two simple features or plain word counts are enough to reach about 99.5% on this dataset.
- Saving the trained models lets the app predict instantly without retraining.
- Both models lean heavily on emoticons such as `:)` and `:(`, which are the strongest signals in the data. Tweets without emoticons would be a tougher test.

## Project structure

```
.
├── Sentiment_LR_NB_Combined.ipynb   # full notebook: training, evaluation and saving the models
├── app.py                           # Streamlit live demo (loads the pickle files)
├── lr_model.pkl                     # saved Logistic Regression model
├── nb_model.pkl                     # saved Naive Bayes model
├── utils.py                         # process_tweet helper (NLTK-based text cleaning)
├── requirements.txt                 # pinned dependencies
└── README.md
```

## Getting started

1. Clone the repository and move into the folder.
2. Use Python 3.11 or newer and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the live demo (the saved models are included, so no training is needed):

   ```bash
   streamlit run app.py
   ```

4. To retrain and recreate the model files, install Jupyter (`pip install jupyter`), open `Sentiment_LR_NB_Combined.ipynb` and run all cells. The NLTK datasets (`twitter_samples` and `stopwords`) are downloaded automatically, and Part D rewrites `lr_model.pkl` and `nb_model.pkl`.

`utils.py`, `lr_model.pkl` and `nb_model.pkl` must stay in the same folder as `app.py`.

## Deploy your own copy (free)

1. Push all files in the project structure above to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, choose your repository and branch, set the main file to `app.py`, and click **Deploy**.

Free apps may sleep after a period of inactivity and wake up on the next visit.

## Acknowledgements

This project is inspired by the Logistic Regression and Naive Bayes exercises from the Natural Language Processing Specialization. The implementation here is rebuilt using pandas and scikit-learn.

## Author

**Alisha Khan**
Add your LinkedIn profile link here.
