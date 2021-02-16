import pandas as pd
import pprint as pp
import os
import re, string
import numpy as np
import math
from collections import Counter
from collections import OrderedDict
import timeit
from collections import defaultdict
from gensim.models import Word2Vec


def preprocess_data(df):
    ''' This function aims to preprocess input dataframe
    
    Parameters
    ----------
    df: dataframe including article Nid, title, description, summary and text
    Return
    ------
    new_df: pd.DataFrame   
    '''
    new_df = df.sort_values(by=['Nid'], inplace=False)
    new_df["text"] = new_df['Title'].fillna('') + '. ' + new_df['Description'].fillna('') + ' ' + new_df['Summary'].fillna('')
    new_df.drop(columns =['Title', 'Description', 'Summary' ], inplace=True)
    return new_df


def process_doc(doc, max_phrase_length=1):
    ''' This function aims to get the part of speech tag count of a words in a given sentence
    
    Parameters
    ----------
    doc: text 
    max_phrase_length : integer to set max length for phrase
    Return
    ------
    new_df: pd.DataFrame   
    '''
    from nltk.stem import WordNetLemmatizer
    import nltk
    from nltk.corpus import wordnet
    from nltk.util import ngrams

    lemmatizer = WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))

    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }

    noun_count = 0
    pron_count = 0
    verb_count = 0
    adj_count = 0
    adv_count = 0

    stopword_count = 0
    upper_case_count = 0
    char_count = 0
    unique_words = set()
    word_count = 0

    tokens = []
    nouns = []
    noun_phrases = []

    sentences = nltk.sent_tokenize(doc)

    for s in sentences:
        words = nltk.word_tokenize(s)
        pairs = nltk.pos_tag(words)

        for pair in pairs:
            tag = list(pair)[1]
            w = list(pair)[0]

            w = re.sub(r'[^a-zA-Z0-9 ]', r'', w)
            w = w.replace('\n', "")
            if w.isdigit(): continue

            if w.isupper(): upper_case_count += 1
            w = w.lower()
            if w in stopwords:
                stopword_count += 1
                continue

            if tag in pos_family['noun']:
                w = lemmatizer.lemmatize(w, 'n')
                if len(w) <= 2: continue

                noun_count += 1

                nouns.append(w)
                tokens.append(w)

                char_count += len(w)
                word_count += 1
                unique_words.add(w)

            elif tag in pos_family['pron']:
                pron_count += 1
                w = lemmatizer.lemmatize(w)
                if len(w) <= 2: continue

                tokens.append(w)

                char_count += len(w)
                word_count += 1
                unique_words.add(w)

            elif tag in pos_family['verb']:
                verb_count += 1
                if len(w) <= 2: continue

                w = lemmatizer.lemmatize(w, 'v')
                tokens.append(w)

                char_count += len(w)
                word_count += 1
                unique_words.add(w)

            elif tag in pos_family['adj']:
                adj_count += 1
                w = lemmatizer.lemmatize(w, 'a')
                if len(w) <= 2: continue

                tokens.append(w)

                char_count += len(w)
                word_count += 1
                unique_words.add(w)

            elif tag in pos_family['adv']:
                adv_count += 1
                w = lemmatizer.lemmatize(w, 'r')
                if len(w) <= 2: continue
                tokens.append(w)

                char_count += len(w)
                word_count += 1
                unique_words.add(w)

    noun_grammar = '''
            NP: {<NN.*|JJ>*<NN.*|VBG.*>}
                {<NNP>+}
            '''
    #noun_grammar = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"

    #Makes chunks using grammar regex
    cp = nltk.RegexpParser(noun_grammar)
    tree = cp.parse(pairs)
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):

        # if subtree.label().startswith('N') or \
        # subtree.label().startswith('P') or \
        # subtree.label().startswith('W'):

        phrases = []
        for w, pos in subtree.leaves():
            w = w.lower()
            if pos.startswith('N'):
                w = lemmatizer.lemmatize(w, 'n')
            elif pos.startswith('J'):
                w = lemmatizer.lemmatize(w, 'a')
            elif pos.startswith('V'):
                w = lemmatizer.lemmatize(w, 'v')

            if len(phrases) < max_phrase_length:
                phrases.append(w)
            else:
                noun_phrases.append('_'.join(phrases))
                phrases = []
                phrases.append(w)

        if len(phrases) > 0:
            noun_phrases.append('_'.join(phrases))

    unique_word_count = len(unique_words)
    word_density = char_count / (word_count + 1)
    sent_count = len(sentences)
    title_word_account = len(doc.split(".")[0].split())

    F = {}
    F['noun_count'] = noun_count
    F['pron_count'] = pron_count
    F['verb_count'] = verb_count
    F['adj_count'] = adj_count
    F['adv_count'] = adv_count

    F['word_count'] = word_count
    F['unique_word_count'] = unique_word_count
    F['word_density'] = word_density
    F['sent_count'] = sent_count
    F['title_word_account'] = title_word_account
    F['upper_case_count'] = upper_case_count
    F['stopword_count'] = stopword_count

    T = {}
    T['tokens'] = ' '.join(tokens)
    T['nouns'] = ' '.join(nouns)
    T['noun_phrases'] = ' '.join(noun_phrases)

    return F, T


def generate_text_features(df, max_phrase_length):
    ''' This function aims to extract text features from input dataframe
    
    Parameters
    ----------
    df: input data 
    max_phrase_length: integer to set max phrase length
    Return
    ------
    X_train: dictionay storing text features  
    '''
    token_list = []
    noun_list = []
    np_list = []

    meta_feature_list = []

    for index, row in enumerate(df['text']):
        F_dict, T_dict = process_doc(row, max_phrase_length)

        token_list.append(T_dict['tokens'])
        noun_list.append(T_dict['nouns'])
        np_list.append(T_dict['noun_phrases'])
        meta_feature_list.append(F_dict)

    meta_feature_df = pd.DataFrame(meta_feature_list, columns=F_dict.keys())

    X_train = {}
    X_train['tokens'] = np.asarray(token_list)
    X_train['nouns'] = np.asarray(noun_list)
    X_train['np'] = np.asarray(np_list)
    X_train['meta_df'] = meta_feature_df

    return X_train


def generate_text_data(WV_INPUT_DATA_PATH, TRAIN_PATH, update, max_phrase_length):
    ''' This function aims to generate input data for building word2vec model
    
    Parameters
    ----------
    WV_INPUT_DATA_PATH: input data path
    TRAIN_PATH: data train path
    update: boolean for update train data
    max_phrase_length: integer to set max phrase length
    Return
    ------
    X_train: dictionay storing text features  
    '''
    df = pd.read_csv(WV_INPUT_DATA_PATH, sep=',', encoding='utf-8')
    X_train = {}

    X_train_fname = TRAIN_PATH + "/X_train"
    if (update==False) and os.path.exists(X_train_fname + "_tokens.npz"):
        X_train['tokens'] = np.load(X_train_fname + "_tokens.npz")["arr_0"]
     
    else:
        df = preprocess_data(df)
        X_train = generate_text_features(df, max_phrase_length)
        np.savez_compressed(X_train_fname + "_tokens", X_train['tokens'])
        X_train['meta_df'].to_csv(X_train_fname + "_meta_df.csv", index=False)

    return X_train


def create_skipgram_wv(X_train, data_type, save_dir, update, dim_size, window_size, min_count):
    ''' This function aims to generate word2vec model
    
    Parameters
    ----------
    wv_input: list of text
    data_type: string to represent data type
    save_dir: path to word2vec model
    update: boolean to update the model
    dim_size: integer dimension size
    window_size: integer Maximum distance between the current and predicted word within a sentence.
    min_count: integer Ignores all words with total frequency lower than this.
    Return
    ------
    model: generated word2vec model
    '''
    docs = X_train[data_type]
    wv_input = []

    for doc in docs:
        wv_input.append(doc.split())

    filename = data_type + "_skipgram." + str(dim_size) + "d.txt"
    model_filename = data_type + "_skipgram." + str(dim_size) + "d.model"

    # train word2vec model (sg=0: CBOW, sg=1:skip-gram)
    if (update == True):
        model = Word2Vec(wv_input, size=dim_size, window=window_size, workers=32, min_count=min_count, sg=1, negative=5)
        # summarize vocabulary size in model
        print("word embedding has been done for %s", data_type)
        print("data type to be embedded: %s", data_type)
        print("vocabulary size to be embedded: {0}".format(len(model.wv.vocab)))
        # save model in ASCII (word2vec) format
        model.wv.save_word2vec_format(save_dir + "/" + filename, binary=False)
        model.save(save_dir + "/" + model_filename)
        
    else:
        model = read_object(save_dir + "/" + model_filename)
        
    return model