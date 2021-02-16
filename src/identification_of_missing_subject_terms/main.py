import params
from optparse import OptionParser
# ----------------------
# import local libraries
from exact_matching import *



def main():

    parser = OptionParser(usage="usage: %prog [options] data_generation_files")
    (options, args) = parser.parse_args()
    if (len(args) < 1):
        parser.error("Must specify the parameter filename...")

    # Read parameters
    # ---------------------------------------------------------------
    pm = params.Param(path=args[0])

    DESCRIPTION_DATA = pm.DESCRIPTION_DATA
    SUBJECT_TERM_DATA = pm.SUBJECT_TERM_DATA
    OUTPUT_DIR = pm.OUTPUT_DIR

    # preprocess description data
    df = pd.read_csv(DESCRIPTION_DATA, sep=',', encoding='utf-8')
    preprocessed_apoDescriptions = preprocess_data(df)
    
    # distinguish uppercase and lowercase subject terms 
    subject_df = pd.read_csv(SUBJECT_TERM_DATA, sep=',', encoding='utf-8')
    upper_subject_terms_df, lower_subject_terms_df = split_subject_terms(subject_df)
    
    # search subject terms in text
    missed_subject_terms_df = identify_missed_subject_terms(preprocessed_apoDescriptions, upper_subject_terms_df, lower_subject_terms_df)
    missed_subject_terms_df.to_csv(OUTPUT_DIR + "missing_subject_terms.csv", sep=',', encoding='utf-8', index=False)
    
    
if __name__ == "__main__":
    main()