import networkx as nx
from collections import Counter

class Ancestor:

    def __init__(self, name, name_idx, ancestor_score=0, subsumer_score=0, depth=1):
        self.name = name
        self.name_idx = name_idx
        self.ancestor_score = ancestor_score
        self.subsumer_score = subsumer_score
        self.depth = depth

    def set_subsumer_score(self, subsumer_score):
        self.subsumer_score = subsumer_score

    def set_depth(self, depth):
        self.depth = depth

    def __str__(self):
        attrs = vars(self)

        print_str = ""
        for item in attrs.items():
            print_str += str(item) + ";"
        return print_str


def comp_cond_prob(p_x, p_y, doc_freq, doc_freq_term_pair):
    '''
        calculate:
            - prob(x|y)
            - prob(y|x)
    '''

    p_xy = doc_freq_term_pair[p_x, p_y] / doc_freq[p_y]
    p_yx = doc_freq_term_pair[p_y, p_x] / doc_freq[p_x]

    return p_xy, p_yx


def get_ancestors(y, unique_term_list, doc_freq, doc_freq_term_pair, theta):
    ''' This function aims to find ancestors
    
    Parameters
    ----------
    y: string subject term
    unique_term_list: list of subject term
    ddoc_freq: dictionary (key: subject term, value: counted occurrence)
    doc_freq_term_pair: dictionary (key: subject term pair, value: counted occurrence)
    theta: float threshold to build taxonomy
    
    Return
    ------
    ances: list of ancestors
    '''
    ances = []
    for x in unique_term_list:
        if x == y: continue
        p_xy, p_yx = comp_cond_prob(x, y, doc_freq, doc_freq_term_pair)

        if p_xy >= theta and p_yx < theta:
            # x is a subsumer of y
            p = Ancestor(x, unique_term_list.index(x), ancestor_score=p_xy, subsumer_score=p_yx, depth=1)
            ances.append(p)

    return ances


def choose_unique_parent(G, ROOT):
    ''' This function aims to select unique parent node
    
    Parameters
    ----------
    G: generated graph
    ROOT: string name of root
    Return
    ------
    G: generated graph with unique parent node
    '''
    edges = nx.bfs_edges(G, ROOT)
    nodes = [ROOT] + [v for u, v in edges]
    for n in nodes:
        ancestor_paths = nx.all_simple_paths(G, source=ROOT, target=n, cutoff=11)
        ancestor_paths = [p for p in ancestor_paths]
	
        if len(ancestor_paths) > 1:
            scores = []
            depths = []
            max_depth = 0
            for path in ancestor_paths:
                reverse_path = path[::-1]

                # Calculate the subsumption score of the path, given node: n
                score = 0
                depth = 1
                for i in range(len(reverse_path) - 1):
                    source = reverse_path[i]
                    target = reverse_path[i + 1]
                    w = G[target][source]['ps']
                    score += (1 / depth) * w
                    #print("\t", n, round(w, 2), round(score, 2), depth)
                    depth += 1
                scores.append(score)
                depths.append(depth)
                max_depth = max(depth, max_depth)
            max_score_index = scores.index(max(scores))

            # save max score path to delete unneccessary edges
            max_score_path = ancestor_paths[max_score_index]
            max_reverse_path = max_score_path[::-1]
            
            for i, s in enumerate(scores):
                if i != max_score_index:
                    reverse_path = ancestor_paths[i][::-1]
                    if G.has_edge(reverse_path[1], n) and reverse_path[1] != max_reverse_path[1]:
                        G.remove_edge(reverse_path[1], n)

    return G


def build_graph(parent_dict, ROOT):
    ''' This function aims to generate taxonomy graph
    
    Parameters
    ----------
    parent_dict: dictionary storing ancestors
    ROOT: string name of root
    Return
    ------
    G: generated graph
    '''
    G = nx.DiGraph()
    for n, parent_list in parent_dict.items():
        if len(parent_list) == 0:
            G.add_edge(ROOT, n, ps=0, ss=0)
        else:
            for p in parent_list:
                G.add_edge(p.name, n, ps=p.ancestor_score, ss=p.subsumer_score)
                
    return G


def gen_subsumptions(ROOT, unique_term_list, doc_freq, doc_freq_term_pair, theta):
    ''' This function aims to generate taxonomy graph
    
    Parameters
    ----------
    ROOT: string name of root
    unique_term_list: list of subject term
    doc_freq: dictionary (key: subject term, value: counted occurrence)
    doc_freq_term_pair: dictionary (key: subject term pair, value: counted occurrence)
    theta: float threshold to build taxonomy
    Return
    ------
    G: generated subject term taxonomy
    '''
    ances_dict = {}
    for t in unique_term_list:
        ances_dict[t] = get_ancestors(t, unique_term_list, doc_freq, doc_freq_term_pair, theta)

    G = build_graph(ances_dict, ROOT)
    # Calculate subsumption scores of all topics, and choose the only one parent for each topic
    G = choose_unique_parent(G, ROOT)
   
    return G


def doc_freq_func(doc_list):
    ''' This function aims to count subject term 
    
    Parameters
    ----------
    doc_list: list of subject terms
    Return
    ------
    doc_freq: dictionary (key: subject term, value: counted occurrence)
    '''
    doc_freq = Counter()
    for ii, d_list in enumerate(doc_list):
        for token in d_list:
            doc_freq[token] += 1

    print('unigram vocabulary size: {}'.format(len(doc_freq)))
    print('most common: {}'.format(doc_freq.most_common(10)))

    return doc_freq


def doc_freq_term_pair_func(doc_list):
    ''' This function aims to count subject term pairs
    
    Parameters
    ----------
    doc_list: list of subject terms
    Return
    ------
    doc_freq_term_pair: dictionary (key: pair of subject term, value: counted occurrence)
    '''
    doc_freq_term_pair = Counter()
    for ii, d_list in enumerate(doc_list):
        for token1 in d_list:
            for token2 in d_list:
                if token1 != token2:
                    doc_freq_term_pair[token1, token2] += 1

    print('skipgram vocabulary size: {}'.format(len(doc_freq_term_pair)))
    print('most common: {}'.format(doc_freq_term_pair.most_common(15)))

    return doc_freq_term_pair


def calculate_depth(G, root):
    ''' This function aims to calculate depth for a given taxonomy
    
    Parameters
    ----------
    G: graph storing subject term taxonomy
    root: string for root node 
    Return
    ------
    max_depth: integer representing maximum depth of taxonomy
    avg_depth: integer representing average depth of taxonomy
    '''
    total_depth = 0
    depths = nx.shortest_path_length(G, root)  # assume a tree rooted at node 0
    max_depth = 0
    for node, depth in depths.items():
        total_depth += depth
        if max_depth < depth:
            max_depth = depth
    avg_depth = round(total_depth/(len(depths.items())-1), 3)

    return max_depth, avg_depth  # exclude ROOT node



def generate_taxonomy(threshold, df, ROOT):
    ''' This function aims to generate a taxonomy by given documents
    
    Parameters
    ----------
    threshold: float threshold to build taxonomy (e.g. 0.2)
    df: dataframe storing refined subject terms
    ROOT: string for root node 
    Return
    ------
    G: graph storing subject term taxonomy
    '''
    from ast import literal_eval
  
    term_in_docs = []
    term_list = []
    for index, row in df.iterrows():
        if len(row['terms']) > 0:
            terms = row['terms']
        else:
            terms = []
        term_in_docs.append(terms)
        for term in terms:
            term_list.append(term)

    unique_term_list = list(dict.fromkeys(term_list)) # remove duplications in list
    
    doc_freq = doc_freq_func(term_in_docs)
    doc_freq_term_pair = doc_freq_term_pair_func(term_in_docs)
    
    G = gen_subsumptions(ROOT, unique_term_list, doc_freq, doc_freq_term_pair, float(threshold))
    print('threshold: ', threshold)
    print('Number of nodes: ', G.number_of_nodes())
    print('Number of edges: ', G.number_of_edges())
    print('AVG depth and MAX depth:', calculate_depth(G, ROOT))
    return G
  
    
    
