# MUD

Author:
 + Pierre-Antoine Jean.
 
Co-authors:
 + Sebastien Harispe
 + Sylvie Ranwez
 + Patrice Bellot
 + Jacky Montmain

MUD (Multiple Uncertainty Detection) allows to detect linguistic uncertainty in natural language. It is based on the statistical analysis of multiple lexical and syntactic features used to characterize sentences through vector-based representations that can be analyzed by proven classification methods.

#### Dependences ####
MUD.py requires:
 + Python 2.7
 + nltk http://www.nltk.org/
 + numpy http://www.numpy.org/
 + sklearn http://scikit-learn.org/stable/

#### Folders ####
Data folder contain WikiWeasel, BioScope and SFU data (after the conversion of the XML format) and various files build by scripts.

#### Scripts description ####
Binomial_law.py allows to calculate the binomial law for lemms, bigrams, trigrams and PoS patterns. Moreover, this script calculates p(U).

Binomial_law.py uses pattern_cue_pos.py to build files lemms, bigrams, trigrams and PoS patterns with their p(w) value.

Then, MUD.py allows to build features matrice, executes the SVM method and print precision, recall and F-mesure in case of evaluation mode.

#### Run ####
Run MUD.py with at least one option:
 + Choose your training corpus
 
	. w = WikiWeasel - Wikipedia articles - (semantic uncertainty and discourse-level uncertainty)
 
	. b = BioScope - biomedical papers - (semantic uncertainty)

 	. sfu = SFU corpus - gereral corpus - (semantic uncertainty)
 + (optional) your file with one sentence per line.

Examples :
 + python MUD.py w path_your_file
 + python MUD.py path_your_file in this case the training corpus by default is SFU.
 + python MUD.py w is the evaluation mode with WikiWeasel.

Your uncertain sentences are in Data/Results file.

#### References ####
 + Un modèle probabiliste pour la détection de l’incertitude dans le langage naturel. Pierre-Antoine Jean, Sebastien Harispe, Sylvie Ranwez, Patrice Bellot, Jacky Montmain. CORIA, 2016.
 + Uncertainty detection in natural language:a probabilistic model. Pierre-Antoine Jean, Sebastien Harispe, Sylvie Ranwez, Patrice Bellot, Jacky Montmain. WIMS, 2016.

