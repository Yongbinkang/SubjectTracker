# SubjectTracker: A Framework for Refining Subject Terms and Inducing Subject Indexing with Taxonomy

## Introduction

In digital repositories, it is crucial to refine existing subject terms and exploit a taxonomy with subject terms, in order to promote information retrieval tasks such as indexing, cataloging and searching of digital documents. SubjectTracker is a framework addressing how to refine an existing set of subject terms, often containing irrelevant ones or creating noise, that are used to index digital documents. Further, it contains the implementation of automatically inducing a subject term taxonomy to capture and utilise the semantic relations among subject terms.

## Setup steps
1. Clone the repository
```
git clone https://github.com/Yongbinkang/SubTermTracker.git
```
2. Install dependencies
```
pip install requirements.txt
```
## Directory structure

We sketch out the directory structure with description below:

* The __`data/`__ directory contains input or output data for the each process. 

* The __`model/`__ directory contains the generated word2vec model with sample data. 

* The __`src/`__ directory contains all required program files. The source files are divided to four subtasks: 
	- identification_of_missing_subject_terms : This directory contains program files for identifying missing subject terms. 
	- generate_word2vec_model : This directory contains program files for generating a word2vec model 
	- filtering_out_irrelevant_subject_terms : This directory contains program files for merging/removing irrelevant subject terms (secondary subject terms)
	- inducing_subject_term_taxonomy : This directory contains program files for inducing a subject term taxonomy
	
## Demo

Instruction for using the demo code of SubTermTracker:
1. Read the sample documents and subject term data to identify missing subject terms.
2. Generate word2vec model with the sample documents.
3. Filltering out irrelevant subject terms. 
4. Inducing a subject term taxonomy with refined subject terms.

All source code directories contain a demo program file (e.g. induce_taxonomy/demo_building_taxonomy.ipynb) which explains how to excute program file and set parameters. Below explanations describe each demo file:

* demo_identifying_missing_subject_terms.ipynb: demonstrate how to identify missing subject terms in the given documents. The program uses Regular Expression to search subject terms in documents and identify the missing subject terms.
* demo_word2vec_generation.ipynb:demonstrate how to generate gensim word2vec model from the sample documents. 
* demo_merging_subject_terms.ipynb: demonstrate how to merge or remove irrelevant subject terms. 
* demo_building_taxonomy.ipynb: demonstrate how to induce subject term taxonomy with the refined subject terms. 

## Citation
```
@article{Kang_dss:2021,
	author = {Kang, Yong-Bin and Woo, Jihoon and Kneebone, Les and Sellis, Timos},
	doi = {https://doi.org/10.1016/j.dss.2021.113542},
	isbn = {0167-9236},
	journal = {Decision Support Systems},
	keywords = {Controlled terms; Subject taxonomies; Subject headings; Thesaurus construction; APO; Digital library},
	title = {Methodology for refining subject terms and supporting subject indexing with taxonomy: A case study of the APO digital repository},
	url = {https://www.sciencedirect.com/science/article/pii/S016792362100052X},
	year = {2021}}
```



