
from optparse import OptionParser
import pandas as pd
import pathlib
import networkx as nx
import pickle
# import local libraries
import params
from generate_taxonomy_input import generate_taxonomy_input_data
from build_taxonomy import *

def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    if (len(args) < 1):
        parser.error("Must specify the parameter filename...")

    # Read parameters
    # -----------------------------------------------------------------------
    pm = params.Param(path=args[0])
    
    DATA_PATH = pm.DATA_PATH
    ROOT = pm.ROOT
    THRESHOLD = float(pm.THRESHOLD)
    MIN_DF = int(pm.MIN_DF)

    input_df = generate_taxonomy_input_data(DATA_PATH, MIN_DF)
    G = generate_taxonomy(THRESHOLD, input_df, ROOT)
    # build subject term taxonomy
    nx.write_gpickle(G, DATA_PATH+"taxonomy_{}.gpickle".format(THRESHOLD))

if __name__ == "__main__":
    main()