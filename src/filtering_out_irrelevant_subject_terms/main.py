import pandas as pd
from optparse import OptionParser

# ----------------------
# import local libraries
import params
from merging_terms_direct import *


def main():

    parser = OptionParser(usage="usage: %prog [options] data_generation_files")
    (options, args) = parser.parse_args()
    if (len(args) < 1):
        parser.error("Must specify the parameter filename...")

    # Read parameters
    # ---------------------------------------------------------------
    pm = params.Param(path=args[0])

    INPUT_DATA_DIR = pm.INPUT_DATA_DIR
    OUTPUT_DIR = pm.OUTPUT_DIR
    MIN_DF = int(pm.MIN_DF)
    APO_MODEL_DIR = pm.APO_MODEL_DIR
    MIN_SIM = float(pm.MIN_SIM)

    # identify subject term's document frequency
    terms_and_doc_freq_df = terms_doc_freq(INPUT_DATA_DIR)

    # update merging result with wordnet merging
    df = merging_terms_with_wordnet(terms_and_doc_freq_df, MIN_DF, MIN_SIM)
    # merging with word2vec model
    updated_df = merging_terms_with_WV_model(df, APO_MODEL_DIR, MIN_DF, MIN_SIM)
    updated_df.to_csv(OUTPUT_DIR + 'merged_subject_terms.csv', sep=',', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
