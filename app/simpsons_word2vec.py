# import re  # For preprocessing
# import pandas as pd  # For data handling
# from time import time  # To time our operations
# from collections import defaultdict  # For word frequency

# import spacy  # For preprocessing
# import en_core_web_sm

from flask import jsonify
import datetime

# import logging  # Setting up the loggings to monitor gensim
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

# df = pd.read_csv('data/simpsons_dataset.csv')
# df.shape

# print(df.head())

# print(df.isnull().sum())

# df = df.dropna().reset_index(drop=True)
# print(df.isnull().sum())

# nlp = en_core_web_sm.load(disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

# def cleaning(doc):
#     # Lemmatizes and removes stopwords
#     # doc needs to be a spacy Doc object
#     txt = [token.lemma_ for token in doc if not token.is_stop]
#     # Word2Vec uses context words to learn the vector representation of a target word,
#     # if a sentence is only one or two words long,
#     # the benefit for the training is very small
#     if len(txt) > 2:
#         return ' '.join(txt)

# brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])

# t = time()

# txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

# print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

# df_clean = pd.DataFrame({'clean': txt})
# df_clean = df_clean.dropna().drop_duplicates()
# df_clean.shape

# from gensim.models.phrases import Phrases, Phraser

# sent = [row.split() for row in df_clean['clean']]
# phrases = Phrases(sent, min_count=30, progress_per=10000)
# bigram = Phraser(phrases)
# sentences = bigram[sent]

# word_freq = defaultdict(int)
# for sent in sentences:
#     for i in sent:
#         word_freq[i] += 1
# len(word_freq)
# sorted(word_freq, key=word_freq.get, reverse=True)[:10]

# import multiprocessing

from gensim.models import Word2Vec

# cores = multiprocessing.cpu_count() # Count the number of cores in a computer
# w2v_model = Word2Vec(min_count=20,
#                      window=2,
#                      size=300,
#                      sample=6e-5, 
#                      alpha=0.03, 
#                      min_alpha=0.0007, 
#                      negative=20,
#                      workers=cores-1)

# t = time()

# w2v_model.build_vocab(sentences, progress_per=10000)

# print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# t = time()

# w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

# print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# w2v_model.init_sims(replace=True)

w2v_model = Word2Vec.load('./data/w2v_simpsons.model')
# print('sdfsfd')
# print(w2v_model.wv.most_similar(positive=["homer"]))

# print(w2v_model.wv.most_similar(positive=["bart"]))

def Getmostsim(request):
    #fetch input from form + loading model
    from_form = request.form['most_similar_to']
    # with open('data/news_train.pkl', 'rb') as f:
    #     news_train = pickle.load(f)
    # with open('models/model.pkl', 'rb') as f:
    #     clf = pickle.load(f)
    # with open('prediction_map.json', 'r') as pred_map:
    #     prediction_map = json.load(pred_map)

    # count_vect = CountVectorizer()
    # tfidf_transformer = TfidfTransformer()
    # cv_fit = count_vect.fit_transform(news_train.data)
    # X_train_tfidf = tfidf_transformer.fit_transform(cv_fit)

    # count_vect_data = count_vect.transform([from_form])
    # tfidf_transformer_data = tfidf_transformer.transform(count_vect_data)
    # prediction = clf.predict(tfidf_transformer_data)
    # prediction_name = prediction_map.get(str(prediction[0]), "couldn't find name")

    print(from_form)
    try:
        most_similar = w2v_model.wv.most_similar(positive=[from_form])
    except KeyError as e: # thrown when searched word not in vocabulary
        return str(e), 400
  
    response = {
        'status': 200,
        'prediction': most_similar,
        'created_at': datetime.datetime.now()
    }
    # print(from_form)
    return jsonify(response)
    # return jsonify(['sdf'])

def similarity(request):
    from_form = request.form['words_to_compare']
    print(from_form)
    words = from_form.split(',')
    print(words)

    try:
        similarity_score = w2v_model.wv.similarity(words[0].strip(), words[1].strip())
    except KeyError as e: # thrown when searched word not in vocabulary
        return str(e), 400
  
    response = {
        'status': 200,
        'prediction': str(similarity_score),
        'created_at': datetime.datetime.now()
    }

    return jsonify(response)
