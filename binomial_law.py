#!/bin/python
# -*- coding: utf-8 -*-

import math, sys
import pattern_cue_pos as pcp

def check_uncertainty_sentence(phrase):
	uncertainty = 0
	for mot in phrase:
		spl = mot.split("\t")
		if spl[len(spl)-1][:-1] != "O":
			uncertainty = 1
	return uncertainty

def number_cue_sentence(phrase):
	uncertainty = 0
	for mot in phrase:
		spl = mot.split("\t")
		if spl[len(spl)-1][:-1] != "O":
			uncertainty += 1
	return uncertainty

def number_cue_sentence_bi_tri(phrase):
	bi_cues = 0
	tri_cues = 0
	cue = 0
	
	for i_mot in range(len(phrase)):
		spl = phrase[i_mot].split("\t")
		if spl[len(spl)-1][:-1] != "O":
			cue += 1
		elif cue >= 2:
			bi_cues += cue - 1
			if cue >= 3:
				tri_cues += cue - 2
			cue = 0

	return bi_cues, tri_cues	
		
def factorial(n, factorial_calculs):
	if n == 0:
		return 1, factorial_calculs
	else:
		#return reduce((lambda x,y: x*y),range(1,n+1))
		resultat = math.factorial(n)
		factorial_calculs[n] = resultat
		return resultat, factorial_calculs
    	
def coef_binomial(n, k, coef_binomial_calculs, factorial_calculs):
	diff = n - k
	
	facto_k = 0
	facto_diff = 0
	facto_n = 0
	
	
	if factorial_calculs.has_key(k):
		facto_k = factorial_calculs[k]
	else:
		facto_k, factorial_calculs = factorial(k, factorial_calculs)
		
	if factorial_calculs.has_key(diff):
		facto_diff = factorial_calculs[diff]
	else:
		facto_diff, factorial_calculs = factorial(diff, factorial_calculs)
		
	if factorial_calculs.has_key(n):
		facto_n = factorial_calculs[n]
	else:
		facto_n, factorial_calculs = factorial(n, factorial_calculs)
	
	
	denominator = facto_k * facto_diff
	#resultat = Decimal(facto_n)/Decimal(denominator)
	resultat = float(facto_n)/float(denominator)

	if coef_binomial_calculs.has_key(n):
		coef_binomial_calculs[n][k] = resultat
	else:
		coef_binomial_calculs[n] = {}
		coef_binomial_calculs[n][k] = resultat
		
	return resultat, coef_binomial_calculs, factorial_calculs

def binomial_law(pu, sw, suw, coef_binomial_calculs, factorial_calculs):
	#pu = Decimal(pu)
	pu = float(pu)
	# We calculate until >= n.
	resultat = 0.0
		
	for i_suw in range(suw, (sw+1)):
		
		new_coef = 0
		if coef_binomial_calculs.has_key(sw):
			if coef_binomial_calculs[sw].has_key(i_suw):
				coef = coef_binomial_calculs[sw][i_suw]
			else:
				new_coef = 1
		else:
			new_coef = 1
		
		if new_coef == 1:
			coef, coef_binomial_calculs, factorial_calculs = coef_binomial(sw, i_suw, coef_binomial_calculs, factorial_calculs)
			
		intermediaire1 = pu**float(i_suw)
		diff = sw - i_suw
		intermediaire2 = (1.0-pu)**diff
		resultat += (coef * intermediaire1 * intermediaire2)
		
				
	return resultat, coef_binomial_calculs, factorial_calculs

# Approximation of the binomial law when n is too high.
def erf(z):
        t = 1.0 / (1.0 + 0.5 * abs(z))
        # use Horner's method
        ans = 1 - t * math.exp( -z*z -  1.26551223 +
                                                t * ( 1.00002368 +
                                                t * ( 0.37409196 + 
                                                t * ( 0.09678418 + 
                                                t * (-0.18628806 + 
                                                t * ( 0.27886807 + 
                                                t * (-1.13520398 + 
                                                t * ( 1.48851587 + 
                                                t * (-0.82215223 + 
                                                t * ( 0.17087277))))))))))
        if z >= 0.0:
                return ans
        else:
                return -ans

# This fonction corresponding directly at 1-b_l.
def normal_estimate(s, p, n):
    u = n * p
    o = (u * (1-p)) ** 0.5

    return 0.5 * (1 + erf((s-u)/(o*2**0.5)))

def read_file_4col(file_in):
	lemm = []
	for ligne in file_in:
		spl = ligne.split(" ")
		if len(spl) == 4:
			lemm.append([spl[0], int(spl[1]), float(spl[2]), int(spl[3])])
	return lemm
	
def read_file_6col(file_in):
	lemm = []
	for ligne in file_in:
		spl = ligne.split(" ")
		if len(spl) == 6:
			lemm.append([spl[0], int(spl[1]), float(spl[2]), int(spl[3]), float(spl[4]), int(spl[5])])
	return lemm

def write_features(name_file_in, name_file_out, lemm_file, pu, pu_su, coef_binomial_calculs, factorial_calculs, nb_mot_global):
	file_in = open(name_file_in, "r").readlines()
	lignes = read_file_6col(file_in)
	file_out = open(name_file_out, "w")
	# lemme - p(w) - bl - p(w)su - blsu - k - ksu - n
	for cel in lignes:
		# cel[2] = p(w) for cue.
		# cel[3] = n
		# cel[1] = k for cues.
		# cel[5] = k for uncertainty sentences.
		
		if cel[3] > 150:
			if cel[1] == 0:
				pc = 0.0
			else:
				pc = normal_estimate(cel[1], pu, cel[3])
			
			if cel[5] == 0:
				pc_su = 0.0
			else:
				pc_su = normal_estimate(cel[5], pu_su, cel[3])
			
		else:
			# Binomial law for cues.
			b_l, coef_binomial_calculs, factorial_calculs = binomial_law(pu, cel[3], cel[1], coef_binomial_calculs, factorial_calculs)
			
			#pc = Decimal(1) - b_l
			# If cel[1] == 1, we calculate P(0<= X <= 9) = 1 so 1 - 1 = 0
			if cel[1] == 0:
				pc = 0.0
			else:
				pc = 1.0 - b_l
				
			#print cel[0]+" "+str(pc)
			# Binomial law for lemms into uncertainty sentences.
			b_l_su, coef_binomial_calculs, factorial_calculs = binomial_law(pu_su, cel[3], cel[5], coef_binomial_calculs, factorial_calculs)
			#pc_su = Decimal(1) - b_l_su
			if cel[5] == 0:
				pc_su = 0.0
			else:
				pc_su = 1.0 - b_l_su
		
		# lemm_file allow to bigram (trigram) to obtain a normal anotation for Su.
		
		if cel[1] > 0 or lemm_file == 0:
			file_out.write(cel[0]+"\t"+str(cel[2])+"\t"+str(pc)+"\t"+str(cel[4])+"\t"+str(pc_su)+"\t"+str(cel[1])+"\t"+str(cel[5])+"\t"+str(cel[3])+"\n")
		else:
			file_out.write(cel[0]+"\t"+str(cel[2])+"\t"+str(pc)+"\t"+str(0.0)+"\t"+str(0.0)+"\t"+str(cel[1])+"\t"+str(0)+"\t"+str(cel[3])+"\n")


def build_stat_binom(corpus):
	if corpus == "" or corpus == "w":
		name_file = "Data/WikiWeasel/lemm_pos_chunk_dep_wiki_train.txt"
	elif corpus == "b":
		name_file = "Data/BioScope/bioscope_train.txt"
	elif corpus == "sfu":
		name_file = "Data/SFU/SFU_train_annot.txt"

	file_in = open(name_file, "r").readlines()

	# We build sentences.
	interm = []
	phrase = []
	for line in file_in:
		if line != "\n":
			phrase.append(line)
		else:
			interm.append(phrase)
			phrase = []

	# Optimize calculs - key : n and value : {k - value}.
	coef_binomial_calculs = {}
	factorial_calculs = {}

	# We calculate values of binomial law for each word.
	pcp.proba_lemm_pos(name_file)
	
	# We calculate p(u).
	nb_mot_global = 0
	nb_mot_incertain = 0
	nb_bigram_incertain = 0
	nb_trigram_incertain = 0

	nb_bigram_global = 0
	nb_trigram_global = 0

	# Cues in certain sentences.
	nb_bigram_cs = 0
	# su for sentence uncertainty.
	nb_mot_incertain_su = 0
	nb_bigram_incertain_su = 0
	nb_trigram_incertain_su = 0

	for phrase in interm:
		nb_mot_global += len(phrase)
	
		if len(phrase) >= 2:
			nb_bigram_global += len(phrase) - 1
		if len(phrase) >= 3:
			nb_trigram_global += len(phrase) - 2
	
		uncertainty = check_uncertainty_sentence(phrase)
		if uncertainty == 1:
			# Cues.
			nb_mot_incertain += number_cue_sentence(phrase)
			bi_tri_incertains = number_cue_sentence_bi_tri(phrase)
			nb_bigram_incertain += bi_tri_incertains[0]
			nb_trigram_incertain += bi_tri_incertains[1]
			# Uncertainty sentences.
			nb_mot_incertain_su += len(phrase)
			if len(phrase) >= 2:
				nb_bigram_incertain_su +=  len(phrase) - 1
			if len(phrase) >= 3:
				nb_trigram_incertain_su += len(phrase) - 2
		else:
			nb_bigram_cs += len(phrase) - 1


	pu = float(nb_mot_incertain)/float(nb_mot_global)
	pu_su = float(nb_mot_incertain_su)/float(nb_mot_global)

	print "LEMMS & PATTERNS :"
	print "P(U) = "+str(pu)
	print "P(U) Su = "+str(pu_su)
	"""
	print "Mot incertain :"+str(nb_mot_incertain)
	print "# su :"+str(nb_mot_incertain_su)
	print "#S :"+str(nb_mot_global)
	"""

	write_features("Data/Probability_words/probability_words.txt", "Data/Features/lemm_probability_binomial.txt", 1, pu, pu_su, coef_binomial_calculs, factorial_calculs, nb_mot_global)
	write_features("Data/Probability_words/pattern.txt", "Data/Features/Pattern_probability_binomial.txt", 1, pu, pu_su, coef_binomial_calculs, factorial_calculs, nb_mot_global)
	# Motif Lemms.
	write_features("Data/Probability_words/probability_motif_lemm.txt", "Data/Features/motif_lemms_probability_binomial.txt", 1, pu, pu_su, coef_binomial_calculs, factorial_calculs, nb_mot_global)

	pu = float(nb_bigram_incertain)/float(nb_bigram_global)
	pu_su = float(nb_bigram_incertain_su)/float(nb_bigram_global)

	print "BIGRAMS :"
	print "P(U) = "+str(pu)
	print "P(U) Su = "+str(pu_su)
	"""
	print "Bigram incertain :"+str(nb_bigram_incertain)
	print "# su :"+str(nb_bigram_incertain_su)
	print "#S :"+str(nb_bigram_global)
	"""

	write_features("Data/Probability_words/probability_bigram.txt", "Data/Features/bigram_probability_binomial.txt", 0, pu, pu_su, coef_binomial_calculs, factorial_calculs, nb_mot_global)


	# Bi-Gram for cues in certain sentences.
	#pu = float(nb_bigram_cs)/float(nb_bigram_global)
	#write_features("Data/Probability_words/probability_bigramCS.txt", "Data/Features/bigramCS_probability_binomial.txt", pu, pu_su, coef_binomial_calculs, factorial_calculs, nb_mot_global)

	pu = float(nb_trigram_incertain)/float(nb_trigram_global)
	pu_su = float(nb_trigram_incertain_su)/float(nb_trigram_global)

	print "TRIGRAMS :"
	print "P(U) = "+str(pu)
	print "P(U) Su = "+str(pu_su)
	"""
	print "Trigram incertain :"+str(nb_trigram_incertain)
	print "# su :"+str(nb_trigram_incertain_su)
	print "#S :"+str(nb_trigram_global)
	"""

	write_features("Data/Probability_words/probability_trigram.txt", "Data/Features/trigram_probability_binomial.txt", 0, pu, pu_su, coef_binomial_calculs, factorial_calculs, nb_mot_global)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
