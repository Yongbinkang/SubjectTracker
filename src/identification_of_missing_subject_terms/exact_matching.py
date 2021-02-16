import re
import pandas as pd
import operator
import logging
import pathlib


def preprocess_data(df):
    ''' This function aims to preprocess input dataframe
    
    Parameters
    ----------
    df: input data (pd.dataframe)
        dataframe including article Nid, title, description, summary and text
    Return
    ------
    new_df: pd.DataFrame   
    '''
    new_df = df.sort_values(by=['Nid'], inplace=False)
    new_df["text"] = new_df['Title'].fillna('') + '. ' + new_df['Description'].fillna('') + ' ' + new_df['Summary'].fillna('')
    new_df.drop(columns =['Title', 'Description', 'Summary' ], inplace=True)
    return new_df


def split_subject_terms(subject_df):
    ''' This function aims to split subject term df to uppercasre and lowercase subject term dfs
    
    Parameters
    ----------
    subject_df: dataframe including Term ID and Term
    Return
    ------
    upper_subject_terms_df: dataframe including uppercase subject terms   
    lower_subject_terms_df: dataframe including lowercase subject terms   
    '''
    subject_df.rename(columns={"Term ID": "id", "Term": "terms"}, inplace=True)
    # create new dfs 
    upper_subject_terms_df = pd.DataFrame(columns = ['id', 'terms'])
    lower_subject_terms_df = pd.DataFrame(columns = ['id', 'terms'])

    for index, row in subject_df.iterrows():
        if row['terms'].isupper():
            upper_subject_terms_df = upper_subject_terms_df.append(row)
        else:
            lower_subject_terms_df = lower_subject_terms_df.append(row)
    
    return upper_subject_terms_df, lower_subject_terms_df


def exact_matching(text, subject_terms):
    ''' This function aims to find and count subject terms in a document
    
    Parameters
    ----------
    text: text
    subject_terms: list of subject terms
    Return
    ------
    term_dic: dictionary containing subject term and their occuerences in the given text
    '''
    term_dic = {}
    for term in subject_terms:
        regex = re.compile(r'\b{}\b'.format(term))
        tag = re.findall(regex, text)
        # if matched exist
        if len(tag) > 0:
            # if the matched words is new
            if tag[0] not in term_dic.keys():
                term_dic[str(tag[0])] = len(tag)
            # if the matched words aleady exist
            else:
                term_dic[str(tag[0])] += len(tag)
    # sort by value
    term_dic = sorted(term_dic.items(), key=lambda x: x[1], reverse=True)

    return term_dic


def uppercase_subject_terms_matching(text, upper_subject_terms_df):
    # matching uppercase subject terms
    subject_terms = upper_subject_terms_df["terms"].tolist()
    return exact_matching(text, subject_terms)


def lowercase_subject_terms_matching(text, lower_subject_terms_df):
    # matching lowercase subject terms
    # convert to text and subject terms to lower case
    subject_terms = lower_subject_terms_df["terms"].tolist()
    subject_terms = map(str.lower,subject_terms)
    return exact_matching(text.lower(), subject_terms)


def identify_missed_subject_terms(df, upper_subject_terms_df, lower_subject_terms_df):
    ''' This function aims to identify missomg subject terms
    
    Parameters
    ----------
    df: dataframe (preprocessed_apoDescriptions)
    upper_subject_terms_df: dataframe containing uppercase subject terms
    lower_subject_terms_df: dataframe containing lowercase subject terms
    Return
    ------
    new_df: dataframe containing Nid, Subject(s), and missigned_subject_term columns
    '''
    
    new_df = pd.DataFrame(columns=['Nid', 'Subject(s)', 'uppercase_subject_terms', 'lowercase_subject_terms'])
    
    for i, row in df.iterrows():
        text = row['text']
        # add return output to df
        new_df = new_df.append({'Nid':row['Nid'],
                                'Subject(s)':row['Subject(s)'],
                                'uppercase_subject_terms':uppercase_subject_terms_matching(text, upper_subject_terms_df),
                                'lowercase_subject_terms':lowercase_subject_terms_matching(text, lower_subject_terms_df),
                              }, ignore_index=True)
        
    new_df['missed_subject_terms'] = ''

    for i, row in new_df.iterrows():
        #generate set to contain all matching output term
        new_term_list = []
        upper_terms = row['uppercase_subject_terms']
        lower_terms = row['lowercase_subject_terms']
        
        # combine matched capital and non-capital subject terms
        for term, freq in upper_terms:
            new_term_list.append(term)
        for term, freq in lower_terms:
            new_term_list.append(term.lower())
        
        # get original terms set
        origin_terms = set()
        origin_terms = str(row['Subject(s)']).replace(', ', ',')
        origin_terms = set(origin_terms.split(','))
        
        for origin_term in origin_terms:
            if origin_term.isupper() != True:
                origin_term = origin_term.lower()
            # remove subject terms in the combined subject term list that already appeared in original terms list
            if origin_term in new_term_list:
                new_term_list.remove(origin_term)
        
        # store list to df'
        new_df['missed_subject_terms'].iloc[i] = new_term_list
    # drop columns
    new_df.drop(columns =['uppercase_subject_terms', 'lowercase_subject_terms'], inplace=True)

    return new_df 

        


    



















