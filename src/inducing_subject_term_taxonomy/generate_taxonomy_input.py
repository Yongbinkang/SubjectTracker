import pandas as pd


def replace_infreq_subjects(primary_df, secondary_df, exact_matching_df):
	''' This function aims to replace infrequent subject term to frequent subject term
    
    Parameters
    ----------
    primary_df: dataframe storing frequent subject terms
    secondary_df: dataframe storing infrequent subject terms
    exact_matching_df: dataframe storing subject term for each document
    Return
    ------
    new_df: dataframe stroing replaced subject terms for each document
    '''
	primary_subjects = primary_df['subject_terms'].tolist()
	
	# read secondary subjects and their matched primary subjects
	subject_pairs = {}
	for index, row in secondary_df.iterrows():
		if str(row['matched_frequent_subject_term']) != 'nan':
			subject_pairs[row['subject_terms']] = row['matched_frequent_subject_term']

	# create new df
	new_df = pd.DataFrame(columns = ['Nid', 'terms'])
	unique_terms = set()
	total_unique_terms = set()
	for index, row in exact_matching_df.iterrows():
	    terms = []
	    replaced_terms = []
	    
	    # get terms in Subject(s) column
	    if row['Subject(s)'] != '':
	        subject_terms = str(row['Subject(s)']).replace(', ', ',')
	        for subject in subject_terms.split(','):
	            if subject.isupper() != True:
	                subject = subject.lower()
	            terms.append(subject)
	    # get terms in missed Subjects column
	    if row['missed_subject_terms'] != '':
	        missed_subjects = eval(row['missed_subject_terms'])
	        #missed_subjects_list = list(missed_subjects)
	        for subject in missed_subjects:
	            terms.append(subject)
	  
	    # replace secondary subject term to a primary subject term
	    for term in terms:
	    	if term in primary_subjects:
		    	replaced_terms.append(term) # frequent
		    	total_unique_terms.add(term)
    		else:	# secondary subjects
    			if term in subject_pairs.keys():
	    			replaced_terms.append(subject_pairs[term])
	    			total_unique_terms.add(subject_pairs[term])
	  
	    unique_terms = list(dict.fromkeys(replaced_terms)) #remove duplications in the list
	    new_df = new_df.append({'Nid':row['Nid'], 'terms':unique_terms}, ignore_index=True)
	
	return new_df
	


def generate_taxonomy_input_data(DATA_PATH, MIN_DF):
	''' This function aims to load all required dataframe to generate input data for inducing subject term taxonomy
    
    Parameters
    ----------
    DATA_PATH: path for input data
    MIN_DF: integer representing minimal document frequency
    Return
    ------
    replaced_df: dataframe stroing replaced subject terms for each document
    '''
	missing_subject_term_df = pd.read_csv(DATA_PATH + 'missing_subject_terms.csv', sep=',', encoding='utf-8')
	merged_subject_term_df = pd.read_csv(DATA_PATH + 'merged_subject_terms.csv', sep=',', encoding='utf-8')
	primary_df = merged_subject_term_df.loc[merged_subject_term_df['doc_freq']>=MIN_DF]
	secondary_df = merged_subject_term_df.loc[merged_subject_term_df['doc_freq']<MIN_DF]
	# replace infrequent subjects to merged target subjects
	replaced_df = replace_infreq_subjects(primary_df, secondary_df, missing_subject_term_df)
	
	return replaced_df
	




