from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet_ic
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import pandas as pd
import numpy as np
import logging
import re
import warnings
import pathlib


warnings.filterwarnings('ignore')
#nltk.download('wordnet_ic')

def terms_doc_freq(INPUT_DATA_DIR):
    ''' This function aims to count subject terms' document frequency
    
    Parameters
    ----------
    INPUT_DATA_DIR: path for input data
    Return
    ------
    new_df: dataframe stroing subject terms and their document frequency  
    '''
    input_df = pd.read_csv(INPUT_DATA_DIR, sep=',', encoding='utf-8')

    all_terms_set = set()
    term_dic= {}
    
    for i, row in input_df.iterrows():
        original_terms = str(row['Subject(s)']).replace(', ', ',')
        original_terms = original_terms.split(',')
        new_terms = eval(row['missed_subject_terms'])
        
        # get all original subjects
        for term in original_terms:
            if term.isupper() != True:
                term = term.lower()
            if term not in all_terms_set:
                term_dic[term] = 1
                all_terms_set.add(term)
            else:
                term_dic[term] += 1
        
        #get all missed subjects
        for term in new_terms:
            if term not in all_terms_set:
                all_terms_set.add(term)
                term_dic[term] = 1
            else:
                term_dic[term] += 1      
            
    # remove nan value in dictionary
    if "nan" in term_dic:            
        del term_dic['nan']            
        
    term_list = []
    freq_list = []
    for term, doc_freq in term_dic.items():
        term_list.append(term)
        freq_list.append(doc_freq)

    new_df = pd.DataFrame(columns = ['subject_terms', 'doc_freq'])
    new_df['subject_terms'] = term_list
    new_df['doc_freq'] = freq_list

    new_df.sort_values(by=['doc_freq'], ascending=False, inplace=True)

    return new_df


def preprocess(words):
    ''' This function aims to removce stop- words and lemmatize a given term
    
    Parameters
    ----------
    words: subject term
    Return
    ------
    token_list: list of tokens for the given subject term 
    word_type: string representing data type
    '''
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }
    word_type = ''
    pairs = nltk.pos_tag(words)
    token_list = []
    for pair in pairs:
        tag = list(pair)[1]
        w = list(pair)[0]
        w = re.sub(r'[^a-zA-Z0-9 ]', r'', w)
        w = w.replace('\n', "")
        if w.isdigit(): continue
        if w.isupper(): w = w.lower()
        if w in stop_words: continue

        if tag in pos_family['noun']:
            w = lemmatizer.lemmatize(w, 'n')
            word_type = 'n'
        elif tag in pos_family['pron']:
            w = lemmatizer.lemmatize(w)
        elif tag in pos_family['verb']:
            w = lemmatizer.lemmatize(w, 'v')
            word_type = 'v'
        elif tag in pos_family['adj']:
            w = lemmatizer.lemmatize(w, 'a')
            word_type = 'a'
        elif tag in pos_family['adv']:
            w = lemmatizer.lemmatize(w, 'r')
            word_type = 'r'
        token_list.append(w)
    
    return token_list, word_type


def get_vec(model, token_list, vector_size):
    ''' This function aims to convert subject term to vector value
    
    Parameters
    ----------
    model: word2vec model
    token_list: list of tokens (subject term)
    vector_size: integer size of vector
    Return
    ------
    vec: converted vector value
    '''
    sum_vector = np.zeros((vector_size,)) #dim size = 100
    length_of_term = 0
    for w in token_list:
        if w in model.wv.vocab:
            vector = np.array(model[w])  # convert to np array is faster.
        else:
            vector = np.zeros((vector_size,), dtype=int) # [0, 0, 0, ...0] length=100
           
        sum_vector += vector
        length_of_term += 1
    vec = [(sum_vector) / length_of_term]
    
    return vec


def measure_similarity(model, vec1, vec2):
    ''' This function aims to calculate similarity between two given vector values
    
    Parameters
    ----------
    model: word2vec model
    vec1: vector value
    vec2: vector value
    Return
    ------
    vec: float representing similarity score
    '''
    try:
        similarity = cosine_similarity(vec1, vec2)[0][0]
        similarity = max(0, similarity) # positive value
    except:
        similarity = 0
    return round(similarity, 3)



def find_max_similarity(APO_model, apo_dict, infreq_term, freq_subject_term_list):
    ''' This function aims to find the most similar frequent subject term for a given infrequent subjec term
    
    Parameters
    ----------
    APO_model: word2vec model
    apo_dict: dictionary (key = subject term, value = vector value)
    infreq_term: list of infrequent subject term
    freq_subject_term_list: list of frequent subject term
    Return
    ------
    matched_term: float the most similar subject term
    max_similarity: float similarity
    '''
    max_similarity = 0.0
    for freq_term in freq_subject_term_list:
        similarity = measure_similarity(APO_model, apo_dict[infreq_term], apo_dict[freq_term])
        if similarity > max_similarity:
            max_similarity = similarity
            max_similar_term = freq_term
    if max_similarity != 0:
        matched_term = max_similar_term
    else:
        matched_term = ''
    return matched_term, round(max_similarity, 2)


def convert_word_to_vec(APO_model, terms_and_doc_freq_df):
    ''' This function aims to read input data to pass to converting vector value
    
    Parameters
    ----------
    APO_model: word2vec model
    terms_and_doc_freq_df: dataframe storing subjct term and their document frequency
    Return
    ------
    apo_dict: dictionary (key = subject term, value = vector value) 
    '''
    apo_dict = {}
 
    for i, row in terms_and_doc_freq_df.iterrows():
        subject_term = row['subject_terms']
        if ',' in subject_term:subject_term = subject_term.replace(',', '') # remove comma
        if '-' in subject_term:subject_term = subject_term.replace('-', '') # remove hypen
       
        tokenized_subject_term = nltk.word_tokenize(subject_term)
        preprocessed_subject_term, word_type = preprocess(tokenized_subject_term)
      
        apo_vec = get_vec(APO_model, preprocessed_subject_term, vector_size=100)
      
        apo_dict[row['subject_terms']] = apo_vec

    return apo_dict



def merging_terms_with_WV_model(terms_and_doc_freq_df, APO_MODEL_DIR, MIN_DF, MIN_SIM):
    ''' This function aims to update subject term merging result with word embedding method
    
    Parameters
    ----------
    terms_and_doc_freq_df: dataframe storing subjct term and their document frequency
    APO_MODEL_DIR: word2vec model
    THRESHOLD: integer set minimum document frequency
    Return
    ------
    new_df: dataframe to store wor2vec merging subject term result 
    '''   
    # Load WORD2VEC MODEL
    APO_model = Word2Vec.load(APO_MODEL_DIR)   
    # extract frequent subject terms
    freq_subject_term_list = list(terms_and_doc_freq_df['subject_terms'].loc[terms_and_doc_freq_df['doc_freq'] >= MIN_DF]) 
    # create a new DF to store result
    new_df = pd.DataFrame(columns=['subject_terms', 'doc_freq'])
    # convert all subject term to vector value
    apo_dict = convert_word_to_vec(APO_model, terms_and_doc_freq_df)

    i = 0
    count = 0
    for index, row in terms_and_doc_freq_df.iterrows():
        if type(row['max_similarity']) != str: # subject term merged by wordnet method
            wordnet_similarity = float(row['max_similarity'])
        else:
            wordnet_similarity = 0
        infreq_term = row['subject_terms']
        matched_subject_term = row['matched_frequent_subject_term']
        max_similarity = wordnet_similarity
        
        if int(row['doc_freq']) < MIN_DF: # infrequent subject term
            word_embedding_matched_term, word_embedding_similarity = find_max_similarity(APO_model, apo_dict, infreq_term, freq_subject_term_list)
            if word_embedding_similarity >= MIN_SIM and wordnet_similarity == 0:
                # update merging with word embedding where no merged subject term found by using wordnet
                matched_subject_term = word_embedding_matched_term
                max_similarity = word_embedding_similarity
                count += 1
            
        new_df = new_df.append({'subject_terms': row['subject_terms'],
                                'doc_freq': row['doc_freq'],
                                'matched_frequent_subject_term': matched_subject_term,
                                'max_similarity': max_similarity
                                }, ignore_index=True)
        if i%500 == 0:
            print("{} of subject term merging process completed".format(str(i)))   
        i += 1    
    
    print("Word embedding merging process completed. {} of subject terms merged with word embedding method".format(count))
    return new_df



def wordnet_matching(infreq_term, freq_subject_term_list):
    ''' This function aims to find the most similar subject term by using wordnet
    
    Parameters
    ----------
    infreq_term: subject term (single length)
    freq_subject_term_list: list of frequent subject terms
    Return
    ------
    max_sim: float representing similarity score
    merged_term: string matched subject term
    ''' 
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    lemmatizer = WordNetLemmatizer()
    merged_term = ''
    max_sim = 0
    if len(wordnet.synsets(infreq_term)) > 0:
        token_list, word_type = preprocess(nltk.word_tokenize(infreq_term))
        try:
            infre_term_syn = wordnet.synset(lemmatizer.lemmatize(token_list[0]) + '.{}.01'.format(word_type))

            for fre_term in freq_subject_term_list:
                if len(wordnet.synsets(fre_term)) > 0 :
                    token_list, word_type = preprocess(nltk.word_tokenize(fre_term))
                    try:
                        fre_term_syn = wordnet.synset(lemmatizer.lemmatize(token_list[0]) + '.{}.01'.format(word_type))
                    except:
                        continue
                    if infre_term_syn.pos() == fre_term_syn.pos() and infre_term_syn.pos() != 's' and infre_term_syn.pos() != 'a':
                        similarity = infre_term_syn.jcn_similarity(fre_term_syn, brown_ic)
                    
                        if similarity > max_sim:
                            max_sim = round(similarity, 2)
                            merged_term = fre_term
        except:
            max_sim = 0
            merged_term = ''

    return max_sim, merged_term


def merging_terms_with_wordnet(df, MIN_DF, MIN_SIM):
    ''' This function aims to merge infrequent subject terms by wordnet
    
    Parameters
    ----------
    df: dataframe storing result of word2vec merging subject term
    MIN_DF: integer set minimum document frequency
    MIN_SIM: float set minimum similarity 
    Return
    ------
    new_df: dataframe to store updated merging subject term result 
    ''' 

    df['matched_term'] = ''
    df['max_similarity'] = ''

    # extract frequent single lenth subject terms
    freq_subject_term_list = []
    for term in list(df['subject_terms'].loc[df['doc_freq'] >= MIN_DF]):
        if ' ' not in term:
            freq_subject_term_list.append(term)
    i = 0
    count = 0
    for index, row in df.iterrows():
        if " " not in row['subject_terms'] and row['doc_freq'] < MIN_DF: # infrequent single length terms
            infreq_term = row['subject_terms']
            max_sim, merged_term = wordnet_matching(infreq_term, freq_subject_term_list)
            if max_sim >= MIN_SIM:
                if max_sim > 1: max_sim = 1           
                df['matched_term'].iloc[i] = merged_term
                df['max_similarity'].iloc[i] = max_sim
                count += 1
        i += 1
    df.rename(columns={'terms':'subject_terms', 'matched_term':'matched_frequent_subject_term'}, inplace=True)               
    print("wordnet merging process completed. {} of subject terms merged with wordnet".format(count))
    return df           
