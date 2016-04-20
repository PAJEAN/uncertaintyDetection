#!/bin/python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import binomial_law as binomial_law
import formatage_sentences as formatage_sentences

# File handling.
def read_file_8col_round(file_in, decimal):
	dico = {}
	for ligne in file_in:
		spl = ligne.split("\t")
		if len(spl) == 8:
			# LEMM -- PROBA CUE, BINOMIAL LAW CUE, PROBA Su, BINOMIAL LAW Su, k cue, k Su.
			dico[spl[0]] = [float(spl[1]), round(float(spl[2]),decimal), float(spl[3]), round(float(spl[4]),decimal), float(spl[5]), float(spl[6]), float(spl[7])]
	return dico

def check_uncertainty_sentence(phrase):
	uncertainty = 0
	for mot in phrase:
		if mot[2] != "O":
			uncertainty = 1
	return uncertainty

def build_sentences(file_in):
	sentences = []
	phrase = []
	for line in file_in:
		if line != "\n":
			spl = line.split("\t")
			if len(spl) >= 3:
				if len(spl) == 3:
					label = "X"
					lemm = spl[2][:-1]
				else:
					label = spl[6][:-1]
					lemm = spl[2]
				# POS, LEMM, LABEL, WORD
				phrase.append([spl[1], lemm, label, spl[0]])
		else:
			sentences.append(phrase)
			phrase = []
	return sentences

# To rebuild a sentence respecting ponctuations.
def print_sentence(phrase):
	sentence = ""
	for i in range(len(phrase)):
		if i+1 < len(phrase) and phrase[i+1][1] in [".", ",", "'"]:
			sentence += phrase[i][3]
		elif i+1 == len(phrase) and phrase[i][1] != ".":
			sentence += phrase[i][3]+"."
		elif i+1 == len(phrase):
			sentence += phrase[i][3]
		else:
			sentence += phrase[i][3]+" "
	return sentence

def print_features(features):
	feature = ""
	for i in features:
		feature += str(i)+" "
	return feature	
	
def number_of_uncertainty_sentence(sentences):
	number = 0
	for phrase in sentences:
		if check_uncertainty_sentence(phrase) == 1:
			number += 1
	return number
	
# modif_pos = 0 -> pos & modif_pos = 1 -> lemm => PoS - Lemm - PoS
def build_pattern(phrase, i_mot, modif_pos):
	str_motif = ""
	if i_mot == 0:
		if len(phrase) > 2:
			str_motif = str("-_-_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0]+"_"+phrase[i_mot+2][0])
		elif len(phrase) == 2:
			str_motif = str("-_-_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0]+"_-")
		else:
			str_motif = str("-_-_"+phrase[i_mot][modif_pos]+"_-_-")
	elif i_mot == 1:
		if len(phrase) > 3:
			str_motif = str("-_"+phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0]+"_"+phrase[i_mot+2][0])
		elif len(phrase) == 3:
			str_motif = str("-_"+phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0]+"_-")
		elif len(phrase) == 2:
			str_motif = str("-_"+phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_-_-")	
	elif i_mot >= 2:
		if (i_mot+2) < len(phrase):
			str_motif = str(phrase[i_mot-2][0]+"_"+phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0]+"_"+phrase[i_mot+2][0])
		elif (i_mot+1) < len(phrase):
			str_motif = str(phrase[i_mot-2][0]+"_"+phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0]+"_-")
		else:
			str_motif = str(phrase[i_mot-2][0]+"_"+phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_-_-")
	return str_motif

def write_features_labels(name_file_out, features, train, sentences, labels = []):
	file_out = open(name_file_out, "w")

	for f in range(len(features)):
		
		for c in range(len(features[f])):
			file_out.write(sentences[f][c][1]+"\t"+sentences[f][c][0]+"\t")
			file_out.write(str(features[f][c]))
			if c < len(features[f])-1:
				file_out.write("\t")
			if c == len(features[f]) - 1:
				if train == 1:
					file_out.write("\t"+str(labels[f])+"\n")
				else:
					file_out.write("\n")			

# Build features for train and test benchmarks or other file. 
def build_features(sentences, train, dico_lemm, dico_bigram, dico_trigram, dico_pattern, bioscope, globalMeasure, localMeasure, bool_f_selection):
	
	# globalMeasure :
	# 1 : PMI
	# 2 : ODDS RATIO
	# 4 : CPM
	# 5 : WLLC
	# 6 : p(c|f) = pI(w)
	# localMeasure :
	# 1 : log(s(w))
	# 2 : 1 - 1/s(w)
	# 3 : binomial law
	# 4 : sigmoide
	# 5 : NULL
	
	if bioscope == 1:
		training_file = "Data/BioScope/bioscope_train.txt"
	elif bioscope == 0:
		training_file = "Data/WikiWeasel/lemm_pos_chunk_dep_wiki_train.txt"
	else:
		training_file = "Data/SFU/SFU_train_annot.txt"
	
	features = []
	labels = []
	
	# Not take into account in sentence's score.
	stop_word = ["of", "the", "in", "a", "an", "for", "at", "as", "on", "then", "I", "you", "he", "she", "it", "its", "we", "they", "his", "her", "my", "myself", "itself", "your",  "yours",  "yourself",  "yourselves",  "hers",  "herself", "himself", "themselves", "our",  "ours",  "ourselves", "but", "their", "theirs", "because", "into", "before", "after", "from", "again", "further", "once", "up", "down", "here", "there", "both", "each", "so", "than", "this", "these",  "those", "do",  "does",  "did",  "doing", "by", "and", "with", "about", "against", "between", "through", "above", "below", "over",  "under", "just"]

	# We use esperance for words without match with training corpus.
	# p(U) & p(Su) are calculated in binomial_law.py and represent respectively the probability to obtain an uncertainty marker or a word of an uncertain sentence.
	# BioScope (p(U) & p(Su)/2 - 0.0097 & 0.2032/2).
	if bioscope == 1:
		esperance = 0.001
		esperance_su = 0.10
	# WikiWeasel (p(U) & p(Su)/2 - 0.031 & 0.294/2).
	elif bioscope == 0:
		esperance = 0.031
		esperance_su = 0.147
	# SFU (p(U) & p(Su)/2 - 0.024 & 0.31/2).
	else:
		esperance = 0.024
		esperance_su = 0.155
	
	######### GLOBAL MEASURE #########
	
	#### Odds Ratio ####
	def odd_ratio(dico):
		# Number of bigram in all corpus.
		hash_W = 0.0
		# Number of words in Incertain class.
		hash_c_incert = 0.0
		# Number of words in incertainty sentences.
		hash_c_su = 0.0
		for key in dico.keys():
			hash_W += float(dico[key][6])
			hash_c_incert += dico[key][4]
			hash_c_su += float(dico[key][5])
			
		# Number of words in Certain Class.
		hash_c_cert = hash_W - hash_c_incert
		hash_c_su_cert = hash_W - hash_c_su
		
		return [hash_W, hash_c_incert, hash_c_cert, hash_c_su, hash_c_su_cert]
	
	if globalMeasure in [2,5]:	
		oddy_pat = odd_ratio(dico_pattern)
		oddy = odd_ratio(dico_lemm)
		oddy_bi = odd_ratio(dico_bigram)
	
	#### CPM (categorical proposal difference) ####
	def dfc_dfcb(phrases, typeUncertainty, unigram = 1):
		# typeUncertainty is a variable to know, if uncertainty is sentence-level or word-level.
		# 1 for sentence-level
		# else for word-level

		mots = {}		
		# To prepare a hashtab with all lemms and a tab associated [0] for uncertain and [1] for certain.
		# dfc = mots[0]
		# dfcb = mots[1]
		
		for phrase in phrases:
			uncertain = 0
			if typeUncertainty == 1:
				uncertain = check_uncertainty_sentence(phrase)
			
			for i_mot in range(len(phrase)):
			
				if typeUncertainty != 1:
					if unigram == 1:
						if phrase[i_mot][2] != "O":
							uncertain = 1
					else:
						if i_mot + 1 < len(phrase):
							if phrase[i_mot][2] != "O" and phrase[i_mot + 1][2] != "O":
								 uncertain = 1
				
				lemm = phrase[i_mot][1]
				ok = 1
				
				# Bigram.
				if unigram != 1:
					if i_mot + 1 < len(phrase):
						lemm = phrase[i_mot][1]+"_"+phrase[i_mot+1][1]
					else:
						ok = 0
				
				if ok == 1:		
					if mots.has_key(lemm):
						if uncertain == 1:
							mots[lemm][0] += 1
						else:
							mots[lemm][1] += 1
					else:
						if uncertain == 1:
							mots[lemm] = [1,0]
						else:
							mots[lemm] = [0,1]
		return mots

	def dfc_dfcb_pat(phrases, typeUncertainty):
		# typeUncertainty is a variable to know, if uncertainty is sentence-level or word-level.
		# 1 for sentence-level
		# else for word-level
		
		mots = {}
		# To prepare a hashtab with all lemms and a tab associated [0] for uncertain and [1] for certain.
		# dfc = mots[0]
		# dfcb = mots[1]
			
		for phrase in phrases:
			uncertain = 0
			if typeUncertainty == 1:
				uncertain = check_uncertainty_sentence(phrase)

			for i_mot in range(len(phrase)):
				
				# 0 for full pos pattern.
				str_motif = build_pattern(phrase, i_mot, 0)
			
				if typeUncertainty != 1:
					if phrase[i_mot][2] != "O":
						uncertain = 1
				# For one sentence, we don't keep multiple occurrence for one pattern.
				if mots.has_key(str_motif):
					if uncertain == 1:
						mots[str_motif][0] += 1
					else:
						mots[str_motif][1] += 1
				else:
					if uncertain == 1:
						mots[str_motif] = [1,0]

					else:
						mots[str_motif] = [0,1]
		return mots
	
	
	file_in_w = open(training_file, "r").readlines()	
	sent_wiki = build_sentences(file_in_w)
	
	# CPM formula.
	dfc_b_pat_4 = dfc_dfcb_pat(sent_wiki, 0)
	dfc_b_pat_10 = dfc_dfcb_pat(sent_wiki, 1)

	dfc_b_1 = dfc_dfcb(sent_wiki, 0)
	dfc_b_2 = dfc_dfcb(sent_wiki, 0, 0)
	dfc_b_7 = dfc_dfcb(sent_wiki, 1)
	
	for phrase in sentences:
			
		# Feature 1 : SOMME (1-pc)*pw  ## pour les lemmes.
		sco_lemm = 0.0
		# Feature 2 : SOMME (1-pc)*pw  ## pour les bigrams.
		sco_bigram = 0.0
		# Feature 3 : Nombre mots.
		nb_mot_f5 = 0
		# Feature 4 : SOMME (1-pc)*pw # pour les lemmes Su.
		sco_lemm_su = 0.0
		# Feature 5 : SOMME (1-pc)*pw # pour les patterns Su.
		sco_pat_su = 0.0
		# Feature 6 : Score du lemme Max (feature 1).
		sco_max_lemm = 0.0
	
		if train == 1:
			label = check_uncertainty_sentence(phrase)
	
		for i_mot in range(len(phrase)):
			
				
			# PATTERN
			# modif_pos = 0 -> pos
			# modif_pos = 1 -> lemm => PoS - Lemm - PoS	
			str_motif = build_pattern(phrase, i_mot, 0)
			
			# Parameters.
			if bioscope == 1:
				p = 0.01
				p_su = 0.203
			else:
				p = 0.03
				p_su = 0.29
			n_deno = 1.0
			
			num_lemm = 1
			#num_lemm = 3 --> word
			lemm_actuel = phrase[i_mot][num_lemm]
			
			# MOTIF PoS.
			interm_f4 = 0.0
			interm_f10 = 0.0
			if dico_pattern.has_key(str_motif):
			
				# ---------------------- Global Measure.
				# PMI.
				if globalMeasure == 1:
					interm_f4 = (np.log10(dico_pattern[str_motif][0]+1) - np.log10(p))
					interm_f10 = (np.log10(dico_pattern[str_motif][2]+1) - np.log10(p_su))
				
				# ODDS RATIO.
				elif globalMeasure == 2:
					# interm_f4.
					s_certW = dico_pattern[str_motif][6] - dico_pattern[str_motif][4]
					if s_certW == 0:
						s_certW += 1
					if dico_pattern[str_motif][4] > 0:
						commun_factor = ( dico_pattern[str_motif][4] * float(s_certW) ) / ( float(oddy_pat[1]) * float(oddy_pat[2]) )
						interm_f4 = np.log10( ((dico_pattern[str_motif][4] / float(oddy_pat[1])) - commun_factor) / ((float(s_certW) / float(oddy_pat[2])) - commun_factor) )
					# interm_f10.
					s_certW = dico_pattern[str_motif][6] - dico_pattern[str_motif][5]
					if s_certW == 0:
						s_certW += 1
					if dico_pattern[str_motif][5] > 0:
						commun_factor = ( dico_pattern[str_motif][5] * float(s_certW) ) / ( float(oddy_pat[3]) * float(oddy_pat[4]) )
						interm_f10 = np.log10( ((dico_pattern[str_motif][5] / float(oddy_pat[3])) - commun_factor) / ((float(s_certW) / float(oddy_pat[4])) - commun_factor) )
				
				# CPM
				elif  globalMeasure == 4:
					if dfc_b_pat_4.has_key(str_motif):
						interm_f4 = (float(dfc_b_pat_4[str_motif][0]) - float(dfc_b_pat_4[str_motif][1])) / (float(dfc_b_pat_4[str_motif][0]) + float(dfc_b_pat_4[str_motif][1]))
					if dfc_b_pat_10.has_key(str_motif):
						interm_f10 = (float(dfc_b_pat_10[str_motif][0]) - float(dfc_b_pat_10[str_motif][1])) / (float(dfc_b_pat_10[str_motif][0]) + float(dfc_b_pat_10[str_motif][1]))
				
				# WLLC
				elif  globalMeasure == 5:
					# interm_f4.
					s_certW = dico_pattern[str_motif][6] - dico_pattern[str_motif][4]
					if s_certW == 0:
						s_certW += 1
					if dico_pattern[str_motif][4] > 0:
						interm_f4 = np.abs( (dico_pattern[str_motif][4] / float(oddy_pat[1])) * ( np.log10( (dico_pattern[str_motif][4] / float(oddy_pat[1])) ) + np.log10( (float(s_certW) / float(oddy_pat[2])) ) ) )
					
					# interm_f10.
					s_certW = dico_pattern[str_motif][6] - dico_pattern[str_motif][5]
					if s_certW == 0:
						s_certW += 1
					if dico_pattern[str_motif][5] > 0:
						interm_f10 = np.abs( (dico_pattern[str_motif][5] / float(oddy_pat[3])) * ( np.log10( (dico_pattern[str_motif][5] / float(oddy_pat[3])) ) + np.log10( (float(s_certW) / float(oddy_pat[4])) ) ) )
				
				# p(c|f)
				else:
					interm_f4 = dico_pattern[str_motif][0]
					interm_f10 = dico_pattern[str_motif][2]
					
				
				# ---------------------- Local Measure.
				# log(s(w))
				if localMeasure == 1:
					sco_pat_su +=  (interm_f10 * np.log10(dico_pattern[str_motif][6]))
					
				# 1 - 1/s(w)
				elif  localMeasure == 2:
					sco_pat_su += (interm_f10 * (1.0 - (1.0/dico_pattern[str_motif][6]**(1.0/n_deno))))
				
				# Binomial Law
				elif localMeasure == 3:
					sco_pat_su += (interm_f10 * dico_pattern[str_motif][3])
				
				# Sigmoid
				elif  localMeasure == 4:
					sco_pat_su +=  (interm_f10 * (-1.1+(2.1/ (1.0+2.5*np.exp( ((-1.0*5.0**(-p_su))*dico_pattern[str_motif][6])) ))))
				
				# NULL
				else:
					sco_pat_su += interm_f10
						
			elif train == 0:
				sco_pat_su += esperance_su
				
	
			# dico_lemm[lemm_actuel][0] : proba cues.
			# dico_lemm[lemm_actuel][1] : bl cues.
			# dico_lemm[lemm_actuel][2] : proba su.
			# dico_lemm[lemm_actuel][3] : bl su.
			# dico_lemm[lemm_actuel][4] : k cues.
			# dico_lemm[lemm_actuel][5] : k su.
			# dico_lemm[lemm_actuel][6] : n.
			
			# LEMME / LEMME SU.
			interm_f1 = 0.0
			interm_f7 = 0.0
			if not phrase[i_mot][num_lemm] in stop_word:
				if dico_lemm.has_key(lemm_actuel):
				
					# ---------------------- Global Measure.
					# PMI.
					if globalMeasure == 1:
						interm_f1 = (np.log10(dico_lemm[lemm_actuel][0]+1) - np.log10(p))
						interm_f7 = (np.log10(dico_lemm[lemm_actuel][2]+1) - np.log10(p_su))
						
					# ODDS RATIO.
					elif globalMeasure == 2:
						# interm_f1
						s_certW = dico_lemm[lemm_actuel][6] - dico_lemm[lemm_actuel][4]
						if s_certW == 0:
							s_certW += 1
						if dico_lemm[lemm_actuel][4] > 0:
							commun_factor = ( dico_lemm[lemm_actuel][4] * float(s_certW) ) / ( float(oddy[1]) * float(oddy[2]) )
							interm_f1 = np.log10( ((dico_lemm[lemm_actuel][4] / float(oddy[1]) ) - commun_factor) / ((float(s_certW) / float(oddy[2])) - commun_factor) )
						
						# interm_f7
						s_certW = dico_lemm[lemm_actuel][6] - dico_lemm[lemm_actuel][5]
						if s_certW == 0:
							s_certW += 1
						if dico_lemm[lemm_actuel][5] > 0:
							commun_factor = ( dico_lemm[lemm_actuel][5] * float(s_certW) ) / ( float(oddy[3]) * float(oddy[4]) ) 
							interm_f7 = np.log10( ((dico_lemm[lemm_actuel][5] / float(oddy[3])) - commun_factor) / ((float(s_certW) / float(oddy[4])) - commun_factor) )
				
					# CPM
					elif globalMeasure == 4:
						if dfc_b_1.has_key(lemm_actuel):
							interm_f1 = (float(dfc_b_1[lemm_actuel][0]) - float(dfc_b_1[lemm_actuel][1])) / (float(dfc_b_1[lemm_actuel][0]) + float(dfc_b_1[lemm_actuel][1]))
						if dfc_b_7.has_key(lemm_actuel):
							interm_f7 = (float(dfc_b_7[lemm_actuel][0]) - float(dfc_b_7[lemm_actuel][1])) / (float(dfc_b_7[lemm_actuel][0]) + float(dfc_b_7[lemm_actuel][1]))
				
					# WLLC
					elif globalMeasure == 5:
						# interm_f1
						s_certW = dico_lemm[lemm_actuel][6] - dico_lemm[lemm_actuel][4]
						if s_certW == 0:
							s_certW += 1
						if dico_lemm[lemm_actuel][4] > 0:
							interm_f1 = np.abs( (dico_lemm[lemm_actuel][4] / float(oddy[1])) * ( np.log10( (dico_lemm[lemm_actuel][4] / float(oddy[1])) ) + np.log10( (float(s_certW) / float(oddy[2])) ) ) )
				
						# interm_f7
						s_certW = dico_lemm[lemm_actuel][6] - dico_lemm[lemm_actuel][5]
						if s_certW == 0:
							s_certW += 1
						if dico_lemm[lemm_actuel][5] > 0:
							interm_f7 = np.abs( (dico_lemm[lemm_actuel][5] / float(oddy[3])) * ( np.log10( (dico_lemm[lemm_actuel][5] / float(oddy[3])) ) + np.log10( (float(s_certW) / float(oddy[4])) ) ) )
				
					# p(c|f)
					else:
						interm_f1 = dico_lemm[lemm_actuel][0]
						interm_f7 = dico_lemm[lemm_actuel][2]
				
									
					# ---------------------- Local Measure.
					# log(s(w))
					if localMeasure == 1:
						sco_lemm += (interm_f1 * np.log10(dico_lemm[lemm_actuel][6]))
						sco_lemm_su += (interm_f7 * np.log10(dico_lemm[lemm_actuel][6]))
						
					# 1 - 1/s(w)
					elif localMeasure == 2:
						sco_lemm += (interm_f1 * (1.0 - (1.0/dico_lemm[lemm_actuel][6]**(1.0/n_deno))))
						sco_lemm_su +=  (interm_f7 * (1.0 - (1.0/dico_lemm[lemm_actuel][6]**(1.0/n_deno))))
						
					# Binomial Law
					elif localMeasure == 3:
						sco_lemm += (interm_f1 * dico_lemm[lemm_actuel][1])
						sco_lemm_su += (interm_f7 * dico_lemm[lemm_actuel][3])
						
					# Sigmoid
					elif localMeasure == 4:
						sco_lemm += (interm_f1 * (-1.1+(2.1/ (1.0+2.5*np.exp( ((-1.0*5.0**(-p))*dico_lemm[lemm_actuel][6])) ))))
						sco_lemm_su +=  (interm_f7 * (-1.1+(2.1/ (1.0+2.5*np.exp( ((-1.0*5.0**(-p_su))*dico_lemm[lemm_actuel][6])) ))))
						
					# NULL
					else:
						sco_lemm += interm_f1
						sco_lemm_su +=  interm_f7
							
					# Max
					if interm_f1 > sco_max_lemm:
						sco_max_lemm = interm_f1
						
						
				elif train == 0:
					sco_lemm += esperance
					sco_lemm_su += esperance_su
					# Max
					if esperance > sco_max_lemm:
						sco_max_lemm = esperance
		
		
			# BIGRAM.
			interm_f2 = 0.0
			if i_mot + 1 < len(phrase):
				lemm_bigram = lemm_actuel+"_"+phrase[i_mot+1][1]
				if dico_bigram.has_key(lemm_bigram):
						
					# ---------------------- Global Measure.
					# PMI.
					if globalMeasure == 1:
						interm_f2 = (np.log10(dico_bigram[lemm_bigram][0]+1) - np.log10(p))
				
					# ODDS RATIO.
					elif  globalMeasure == 2:
						# interm_f2
						s_certW = dico_bigram[lemm_bigram][6] - dico_bigram[lemm_bigram][4]
						if s_certW == 0:
							s_certW += 1
						if dico_bigram[lemm_bigram][4] > 0:
							commun_factor = ( dico_bigram[lemm_bigram][4] * float(s_certW) ) / ( float(oddy_bi[1]) * float(oddy_bi[2]) )
							interm_f2 = np.log10( ((dico_bigram[lemm_bigram][4] / float(oddy_bi[1])) - commun_factor) / ((float(s_certW) / float(oddy_bi[2])) - commun_factor) )
				
					# CPM
					elif globalMeasure == 4:
						if dfc_b_2.has_key(lemm_bigram):
							interm_f2 = (float(dfc_b_2[lemm_bigram][0]) - float(dfc_b_2[lemm_bigram][1])) / (float(dfc_b_2[lemm_bigram][0]) + float(dfc_b_2[lemm_bigram][1]))
				
					# WLLC
					elif globalMeasure == 5:
						# interm_f2
						s_certW = dico_bigram[lemm_bigram][6] - dico_bigram[lemm_bigram][4]
						if s_certW == 0:
							s_certW += 1
						if dico_bigram[lemm_bigram][4] > 0:
							interm_f2 = np.abs( (dico_bigram[lemm_bigram][4] / float(oddy_bi[1])) * ( np.log10( (dico_bigram[lemm_bigram][4] / float(oddy_bi[1])) ) + np.log10( (float(s_certW) / float(oddy_bi[2])) ) ) )
				
					# p(c|f)
					else:
						interm_f2 = dico_bigram[lemm_bigram][0]
										
					
					# ---------------------- Local Measure.
					# log(s(w))
					if localMeasure == 1:
						sco_bigram += (interm_f2 * np.log10(dico_bigram[lemm_bigram][6]))
					
					# 1 - 1/s(w)
					elif  localMeasure == 2:
						sco_bigram += (interm_f2 * (1.0 - (1.0/dico_bigram[lemm_bigram][6]**(1.0/n_deno))))
				
					# Binomial Law
					elif localMeasure == 3:
						sco_bigram += (interm_f2 * dico_bigram[lemm_bigram][1])
				
					# Sigmoid
					elif localMeasure == 4:
						sco_bigram += (interm_f2 * (-1.1+(2.1/ (1.0+2.5*np.exp( ((-1.0*5.0**(-p))*dico_bigram[lemm_bigram][6])) ))))
				
					# NULL
					else:
						sco_bigram += interm_f2
					
				elif train == 0:
					sco_bigram += esperance
			
			# NOMBRE MOTS.
			nb_mot_f5 += 1
			
		if nb_mot_f5 == 0:
			nb_mot_f5 = 1
		
		if bool_f_selection == 1:
			features.append([sco_lemm, sco_bigram, nb_mot_f5, sco_lemm_su, sco_pat_su, sco_max_lemm])
		else:
			if bioscope == 0:
				features.append([sco_lemm, sco_bigram, nb_mot_f5, sco_lemm_su, sco_pat_su, sco_max_lemm])
			elif bioscope == 1:
				features.append([sco_lemm, sco_lemm_su, sco_max_lemm])
			else:
				features.append([sco_lemm, sco_lemm_su, sco_max_lemm])
		
		if train == 1:
			labels.append(label)
		
	return features, labels

################################################################## PROGRAM ##################################################################

def MUD(globalMeasure, localMeasure, bool_f_selection = 1):
	
	if len(sys.argv) <= 1:
		print "Parameter require (w, b, sfu) and your file."
		exit()
	# Variable to modify inputs (0 for WikiWeasel and 1 for BioScope).
	# SFU is the training corpus by default.
	if sys.argv[1] == "w":
		bioscope = 0
	elif sys.argv[1] == "b":
		bioscope = 1
	elif sys.argv[1] == "sfu":
		bioscope = 2
	else:
		bioscope = 3
	
	# Binomial law and statistic on training corpus.
	memory = open("Data/Parameter/training_memory.txt", "r").readlines()
	if not sys.argv[1] in ["w", "b", "sfu"]:
		binomial_law.build_stat_binom("sfu")
	else:
		if memory[0][0] != sys.argv[1]:
			if bioscope == 0:
				binomial_law.build_stat_binom("w")
			elif bioscope == 1:
				binomial_law.build_stat_binom("b")
			else:
				binomial_law.build_stat_binom("sfu")
			out = open("Data/Parameter/training_memory.txt", "w")
			out.write(sys.argv[1])
	
	print ""
	
	if globalMeasure == 1:
		print "\t----- Global Measure : PMI -----"
	elif  globalMeasure == 2:
		print "\t----- Global Measure : ODDS RATIO -----"
	elif  globalMeasure == 4:
		print "\t----- Global Measure : CPM -----"
	elif  globalMeasure == 5:
		print "\t----- Global Measure : WLLC -----"
	else:
		print "\t----- Global Measure : p(c|f) -----"
		
	if localMeasure == 1:
		print "\t----- Local Measure : log(s(w)) -----"
	elif  localMeasure == 2:
		print "\t----- Local Measure : 1 - 1/s(w) -----"
	elif localMeasure == 3:
		print "\t----- Local Measure : Binomial Law -----"
	elif  localMeasure == 4:
		print "\t----- Local Measure : Sigmoid -----"
	else:
		print "\t----- Local Measure : NULL -----"
		
	if bioscope == 1:
		print "\t----- BioScope -----"
	elif bioscope == 0:
		print "\t----- WikiWeasel -----"
	else:
		print "\t----- SFU -----"
	
	print ""
	
	# We build four dictionary containing scores of 1-(binomial_law) and p(w).
	file_in = open("Data/Features/Pattern_probability_binomial.txt").readlines()
	dico_pattern = read_file_8col_round(file_in, 5)

	file_in = open("Data/Features/lemm_probability_binomial.txt").readlines()
	dico_lemm = read_file_8col_round(file_in, 5)

	file_in = open("Data/Features/bigram_probability_binomial.txt").readlines()
	dico_bigram = read_file_8col_round(file_in, 5)

	file_in = open("Data/Features/trigram_probability_binomial.txt").readlines()
	dico_trigram = read_file_8col_round(file_in, 5)


	print "----- TRAINING"
	# TRAINING - BUILD MODEL.
	if bioscope == 0:
		file_in = open("Data/WikiWeasel/lemm_pos_chunk_dep_wiki_train.txt", "r").readlines()
	elif bioscope == 1:
		file_in = open("Data/BioScope/bioscope_train.txt", "r").readlines()
	else:
		file_in = open("Data/SFU/SFU_train_annot.txt", "r").readlines()
		
	sentences = build_sentences(file_in)
	features, labels_training = build_features(sentences, 1, dico_lemm, dico_bigram, dico_trigram, dico_pattern, bioscope, globalMeasure, localMeasure, bool_f_selection)

	# Features selection.
	def selection_features_boolean(features, choose):
		f_tot = []
		f = []
		for i in features:
			f = []
			for j in range(len(i)):
				if choose[j] == True:
					f.append(i[j])
			f_tot.append(f)
		return f_tot

	def selection_features(features, TF_features):
		clf = RandomForestClassifier(n_estimators=400)
		clf.fit(features, labels_training)
		print clf.feature_importances_
		print TF_features
		print "\n"
		boolean = []
		new_TF = []
		bad_vector = 0
		i_feat = 0
		for i in clf.feature_importances_:
			if i < 0.08:
				boolean.append(False)
				bad_vector = 1
			else:
				boolean.append(True)
				new_TF.append(TF_features[i_feat])
			i_feat += 1
		features = selection_features_boolean(features, boolean)
		if bad_vector == 1:
			features, new_TF = selection_features(features, new_TF)
		return features, new_TF

	if bool_f_selection == 1:
		TF_features = []
		for i in range(len(features[0])):
			TF_features.append(i)
		print str(len(TF_features))+" features at beginning.\n"

		features_b, new_TF = selection_features(features, TF_features)

		boolean = []
		for i in range(len(TF_features)):
			if i in new_TF:
				boolean.append(True)
			else:
				boolean.append(False)

		features = selection_features_boolean(features, boolean)
		print str(len(features[0]))+" features selected.\n"

	# Learning.
	clf = svm.SVC(C=10, kernel = 'rbf', gamma = 0.0625)
	clf.fit(features, labels_training)
	
	
		
	print "----- PREDICTION"
	
	if len(sys.argv) == 3:
		print "File "+sys.argv[2]
		bioscope = 3
	else:
		if sys.argv[1] == "w":
			bioscope = 0
		elif sys.argv[1] == "b":
			bioscope = 1
		elif sys.argv[1] == "sfu":
			bioscope = 2
		else:
			print "File "+sys.argv[1]
			bioscope = 3
	
	if bioscope == 0:
		file_in = open("Data/WikiWeasel/lemm_pos_chunk_dep_wiki_eval.txt", "r").readlines()
	elif bioscope == 1:
		file_in = open("Data/BioScope/bioscope_eval.txt", "r").readlines()
	elif bioscope == 2:
		file_in = open("Data/SFU/SFU_eval_annot.txt", "r").readlines()
	else:
		if len(sys.argv) == 3:
			path_file = sys.argv[2]
		else:
			path_file = sys.argv[1]
		
		formatage_sentences.formater_phrase(path_file)
		file_in = open("Data/Inputs/sentences.txt", "r").readlines()
	
	# PREDICTION.		
	sentences = build_sentences(file_in)
	features, labels = build_features(sentences, 0, dico_lemm, dico_bigram, dico_trigram, dico_pattern, bioscope, globalMeasure, localMeasure, bool_f_selection)

	# Features selection.
	if bool_f_selection == 1:
		features = selection_features_boolean(features, boolean)

	predictions = clf.predict(features)

	j = 0
	k = 0
	for i in predictions:
		if i == 1:
			j+=1
		else:
			k+=1
		
	print "\n----- RESULTS"
	print "Number of uncertainty sentences "+str(j)
	print "Number of certainty sentences "+str(k)
	
	# Uncertainty sentences.
	print "Total of sentences "+str(len(sentences))
	#print len(predictions)
	out = open("Data/Results/uncertainty_sentences.txt", "w")
	for i in range(len(predictions)):
		if predictions[i] == 1:
			out.write(str(i)+" "+print_sentence(sentences[i])+"\n")
	
	# Evaluation.
	if bioscope != 3:
		labels_eval = []
		for phrase in sentences:
			if check_uncertainty_sentence(phrase) == 1:
				labels_eval.append(1)
			else:
				labels_eval.append(0)

	
		VP = 0
		FN = 0
		FP = 0
		VN = 0

		file_VP = open("Data/VP_FP_FN/VP.txt", "w")
		file_FP = open("Data/VP_FP_FN/FP.txt", "w")
		file_FN = open("Data/VP_FP_FN/FN.txt", "w")
		file_ALL = open("Data/VP_FP_FN/ALL_labels.txt", "w")

		for i in range(len(labels_eval)):
			if labels_eval[i] == 1 and predictions[i] == 1:
				VP += 1
				file_VP.write(print_sentence(sentences[i])+"\n")
				file_VP.write(print_features(features[i]))
				file_VP.write("\n\n")
			elif labels_eval[i] == 1 and predictions[i] == 0:
				FN += 1
				file_FN.write(print_sentence(sentences[i])+"\n")
				file_FN.write(print_features(features[i]))
				file_FN.write("\n\n")
			elif labels_eval[i] == 0 and predictions[i] == 1:
				FP += 1
				file_FP.write(print_sentence(sentences[i])+"\n")
				file_FP.write(print_features(features[i]))
				file_FP.write("\n\n")
			else:
				VN += 1
			file_ALL.write(print_sentence(sentences[i])+"\n")
			file_ALL.write(print_features(features[i])+str(labels_eval[i]))
			file_ALL.write("\n\n")

		#print "VP: "+str(VP)+", FP: "+str(FP)+", FN: "+str(FN)+" , VN: "+str(VN)

		print "VP : "+str(VP)+" (/"+str(number_of_uncertainty_sentence(sentences))+")"
		print "FP : "+str(FP)
		print "FN : "+str(FN)
	
		if VP == 0:
			VP += 1
	
		precision = VP/float(VP+FP)
		print "Precision : "+str(precision)
		rappel = (VP/float(VP+FN))
		print "Rappel : "+str(rappel)
		fmesure = 2.0 * ((precision*rappel)/(precision+rappel))
		print "F-mesure : "+str(fmesure)


# Run MUD.

# globalMeasure :
# 1 : PMI
# 2 : ODDS RATIO
# 4 : CPM
# 5 : WLLC
# 6 : p(c|f) = pI(w)
# localMeasure :
# 1 : log(s(w))
# 2 : 1 - 1/s(w)
# 3 : binomial law
# 4 : sigmoide
# 5 : NULL

# globalMeasure, localMeasure, selectionFeature.
MUD(6,2,0)


		


















