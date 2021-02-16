import pandas as pd
from optparse import OptionParser
# ----------------------
# import local libraries
import params
from word2vec_generation import *

def main():

    parser = OptionParser(usage="usage: %prog [options] data_generation_files")
    (options, args) = parser.parse_args()
    if (len(args) < 1):
        parser.error("Must specify the parameter filename...")

    # Read parameters
    # ---------------------------------------------------------------
    pm = params.Param(path=args[0])

    WV_INPUT_DATA_PATH = pm.WV_INPUT_DATA_PATH
    DATA_TYPE = pm.DATA_TYPE
    TRAIN_PATH = pm.TRAIN_PATH
    MAX_PHRASE_LENGTH = int(pm.MAX_PHRASE_LENGTH)
    DIM_SIZE = int(pm.DIM_SIZE)
    WINDOW_SIZE = int(pm.WINDOW_SIZE)
    MIN_COUNT = int(pm.MIN_COUNT)
    
    # create input data
    X_train = generate_text_data(WV_INPUT_DATA_PATH, TRAIN_PATH, update=True, max_phrase_length=MAX_PHRASE_LENGTH)
    
    # generate word2vec model
    wv_model = create_skipgram_wv(X_train, DATA_TYPE, TRAIN_PATH, update=True, dim_size=DIM_SIZE, window_size=WINDOW_SIZE, min_count=MIN_COUNT)
    
    
    
if __name__ == "__main__":
    main()
