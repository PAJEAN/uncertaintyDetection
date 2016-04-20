#!/bin/python
# -*- coding: utf-8 -*-

def proba_lemm_pos(name_file):
	file_in = open(name_file, "r").readlines()

	# Conversion to sentences
	sentences = []
	phrase = []
	for line in file_in:
		if line != "\n":
			spl = line.split("\t")
			if len(spl) >= 4:
				# POS, LABEL, LEMM, CHUNK, word
				# it PRP it chunk O
				phrase.append([spl[1], spl[6][:-1], spl[2], spl[3], spl[0]])
		else:
			sentences.append(phrase)
			phrase = []
	print "Number of sentences for the training step "+str(len(sentences))
	
	def uncertainty_sentence(phrase):
		uncertainty = 0
		for mot in phrase:
			if mot[1] != "O":
				uncertainty = 1
		return uncertainty
	
	# To stock uncertain patterns.
	motif_all = {}
	# To stock all partterns.
	motif = {}
	# chunks for each pattern.
	chunk = {}
	# All pos in files.
	pos_entier = {}
	# All pos used for uncertainty cues.
	pos_use = {}
	# lemm for uncertainty cues.
	lemm_use = {}
	lemm_entier = {}
	
	# Bi-Gram.
	bigram = {}
	bigram_use = {}
	
	# Bi-Gram for cues in certain sentences.
	#bigram_cs = {}
	
	# Tri-Gram
	trigram = {}
	trigram_use = {}
	
	# Motif window [-1,1] with lemms.
	motif_lemm = {}
	motif_lemm_use = {}
	motif_lemm_su = {}
	
	# Dictionary for lemm, bigram, trigram and PoS motif belonging to Su.
	lemm_su = {}
	bigram_su = {}
	trigram_su = {}
	motif_su = {}
	
	# Test for Bi-Gram cues cs.
	all_cues = {}
	for phrase in sentences:
		for i_mot in range(len(phrase)):
			if phrase[i_mot][1] == "B":
				if all_cues.has_key(phrase[i_mot][2]):
					all_cues[phrase[i_mot][2]] += 1
				else:
					all_cues[phrase[i_mot][2]] = 1
	num_lemm = 2
	#num_lemm = 4  --> word
	for phrase in sentences:
		value_uncertainty = uncertainty_sentence(phrase)
		
		for i_mot in range(len(phrase)):

			# All pos tags of this file.
			if pos_entier.has_key(phrase[i_mot][0]):
					pos_entier[phrase[i_mot][0]] += 1
			else:
				pos_entier[phrase[i_mot][0]] = 1
		
			# All word of this file.
			if lemm_entier.has_key(phrase[i_mot][num_lemm]):
					lemm_entier[phrase[i_mot][num_lemm]] += 1
			else:
				lemm_entier[phrase[i_mot][num_lemm]] = 1
				
			
			# We stock all bi-gram.
			if i_mot + 1 < len(phrase):
				lemm_bigram = phrase[i_mot][2]+"_"+phrase[i_mot+1][2]
				if bigram.has_key(lemm_bigram):
					bigram[lemm_bigram] += 1
				else:
					bigram[lemm_bigram] = 1
					
			# We stock all bi-gram.
			if i_mot + 2 < len(phrase):
				lemm_trigram = phrase[i_mot][2]+"_"+phrase[i_mot+1][2]+"_"+phrase[i_mot+2][2]
				if trigram.has_key(lemm_trigram):
					trigram[lemm_trigram] += 1
				else:
					trigram[lemm_trigram] = 1
			
			#str_motif = ""
			#if not phrase[i_mot][0] in [".", ",", "``", ":"]:
			# the tag (phrase[i_mot]) is always at the position 2.
			# If modif_pos == 0 then motif full pos else if modif_pos == 2 then motif pos + lemm central
			modif_pos = 0
				
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
			
			if motif_all.has_key(str_motif):
				motif_all[str_motif] += 1
			else:
				motif_all[str_motif] = 1
			
			# Motif lemms.
			"""
			if i_mot == 0:
				if len(phrase) > 1:
					str_motif_lemm = str("-_"+phrase[i_mot][2]+"_"+phrase[i_mot+1][2])
				else:
					str_motif_lemm = str("-_"+phrase[i_mot][2]+"_-")
			elif i_mot == 1:
				if len(phrase) >= 3:
					str_motif_lemm = str(phrase[i_mot-1][2]+"_"+phrase[i_mot][2]+"_"+phrase[i_mot+1][2])
				elif len(phrase) == 2:
					str_motif_lemm = str(phrase[i_mot-1][2]+"_"+phrase[i_mot][2]+"_-")	
			elif i_mot >= 2:
				if (i_mot+1) < len(phrase):
					str_motif_lemm = str(phrase[i_mot-1][2]+"_"+phrase[i_mot][2]+"_"+phrase[i_mot+1][2])
				else:
					str_motif_lemm = str(phrase[i_mot-1][2]+"_"+phrase[i_mot][2]+"_-")
			"""
			# 2 for pos_lemm_pos pattern.
			modif_pos = 2
			if i_mot == 0:
				if len(phrase) > 1:
					str_motif_lemm = str("-_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0])
				else:
					str_motif_lemm = str("-_"+phrase[i_mot][modif_pos]+"_-")
			elif i_mot == 1:
				if len(phrase) >= 3:
					str_motif_lemm = str(phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0])
				elif len(phrase) == 2:
					str_motif_lemm = str(phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_-")	
			elif i_mot >= 2:
				if (i_mot+1) < len(phrase):
					str_motif_lemm = str(phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_"+phrase[i_mot+1][0])
				else:
					str_motif_lemm = str(phrase[i_mot-1][0]+"_"+phrase[i_mot][modif_pos]+"_-")
			
			if motif_lemm.has_key(str_motif_lemm):
				motif_lemm[str_motif_lemm] += 1
			else:
				motif_lemm[str_motif_lemm] = 1
			
			# To calculate the number of occurence for an lemm into a uncertainty sentence.
			if value_uncertainty == 1:
				# Lemm.
				if lemm_su.has_key(phrase[i_mot][num_lemm]):
					lemm_su[phrase[i_mot][num_lemm]] += 1
				else:
					lemm_su[phrase[i_mot][num_lemm]] = 1
				# Motif PoS
				if motif_su.has_key(str_motif):
					motif_su[str_motif] += 1
				else:
					motif_su[str_motif] = 1
				# Motif lemms.
				if motif_lemm_su.has_key(str_motif_lemm):
					motif_lemm_su[str_motif_lemm] += 1
				else:
					motif_lemm_su[str_motif_lemm] = 1
				# Bigram.
				if i_mot + 1 < len(phrase):
					lemm_bigram = phrase[i_mot][2]+"_"+phrase[i_mot+1][2]
					if bigram_su.has_key(lemm_bigram):
						bigram_su[lemm_bigram] += 1
					else:
						bigram_su[lemm_bigram] = 1
				# Trigram.
				if i_mot + 2 < len(phrase):
					lemm_trigram = phrase[i_mot][2]+"_"+phrase[i_mot+1][2]+"_"+phrase[i_mot+2][2]
					if trigram_su.has_key(lemm_trigram):
						trigram_su[lemm_trigram] += 1
					else:
						trigram_su[lemm_trigram] = 1
			else:
				# If the sentence is certain we can check if it contain one uncertainty cue and build a bigram with the next word.
				# Perhaps, we can avoid some FP with this analysis.
				if i_mot + 1 < len(phrase):
					lemm_bigram = phrase[i_mot][2]+"_"+phrase[i_mot+1][2]
					"""
					if all_cues.has_key(phrase[i_mot][2]):
						if bigram_cs.has_key(lemm_bigram):
							bigram_cs[lemm_bigram] += 1
						else:
							bigram_cs[lemm_bigram] = 1
					"""
			# To calculate the number of occurence for an uncertainty cue.
			if phrase[i_mot][1] == "B" or phrase[i_mot][1] == "I":
				# I note uncertainty cues (their lemm).
				if lemm_use.has_key(phrase[i_mot][num_lemm]):
					lemm_use[phrase[i_mot][num_lemm]] += 1
				else:
					lemm_use[phrase[i_mot][num_lemm]] = 1
				# Only pos tag use as uncertainty cue.
				if pos_use.has_key(phrase[i_mot][0]):
					pos_use[phrase[i_mot][0]] += 1
				else:
					pos_use[phrase[i_mot][0]] = 1
				# Motif PoS
				if motif.has_key(str_motif):
					motif[str_motif] += 1
				else:
					motif[str_motif] = 1
				# Motif Lemms.
				if motif_lemm_use.has_key(str_motif_lemm):
					motif_lemm_use[str_motif_lemm] += 1
				else:
					motif_lemm_use[str_motif_lemm] = 1
				# Bigram.
				if i_mot + 1 < len(phrase) and phrase[i_mot+1][1] != "O":
					lemm_bigram = phrase[i_mot][2]+"_"+phrase[i_mot+1][2]
					if bigram_use.has_key(lemm_bigram):
						bigram_use[lemm_bigram] += 1
					else:
						bigram_use[lemm_bigram] = 1
				# Trigram.
				if i_mot + 2 < len(phrase) and phrase[i_mot+1][1] != "O" and phrase[i_mot+2][1] != "O":
					lemm_trigram = phrase[i_mot][2]+"_"+phrase[i_mot+1][2]+"_"+phrase[i_mot+2][2]
					if trigram_use.has_key(lemm_trigram):
						trigram_use[lemm_trigram] += 1
					else:
						trigram_use[lemm_trigram] = 1
				
	# I write results in files.
	"""
	file_out = open("pos_use.txt", "w")

	for key in pos_use.keys():
		file_out.write(key+" "+str(pos_use[key])+" / "+str(pos_entier[key])+"\n")
	for p in pos_entier.keys():
		if not pos_use.has_key(p):
			file_out.write(p+" 0 / "+str(pos_entier[p])+"\n")

	"""

	def write_file(dic_all, dic_use, dic_su, name_file):
		file_out = open(name_file, "w")

		for key in dic_all.keys():
			if dic_use.has_key(key):
				# For cues.
				conf = float(dic_use[key]) / float(dic_all[key])
				# For lemms into uncertainty sentences.
				if dic_su.has_key(key):
					conf_su = float(dic_su[key]) / float(dic_all[key])
					str_dic_su = dic_su[key]
				else:
					conf_su = 0.0
					str_dic_su = 0
				str_dic_use = dic_use[key]
			else:
				conf = 0.0
				str_dic_use = 0
				if dic_su.has_key(key):
					conf_su = float(dic_su[key]) / float(dic_all[key])
					str_dic_su = dic_su[key]
				else:
					conf_su = 0.0
					str_dic_su = 0
				
				
			file_out.write(key+" "+str(str_dic_use)+" "+str(conf)+" "+str(dic_all[key])+" "+str(conf_su)+" "+str(str_dic_su)+"\n")
			
			
	write_file(motif_all, motif, motif_su, "Data/Probability_words/pattern.txt")
	write_file(lemm_entier, lemm_use, lemm_su, "Data/Probability_words/probability_words.txt")
	write_file(bigram, bigram_use, bigram_su, "Data/Probability_words/probability_bigram.txt")
	write_file(trigram, trigram_use, trigram_su, "Data/Probability_words/probability_trigram.txt")

	#write_file(bigram, bigram_cs, {"keyloog":1.0}, "Data/Probability_words/probability_bigramCS.txt")
	write_file(motif_lemm, motif_lemm_use, motif_lemm_su, "Data/Probability_words/probability_motif_lemm.txt")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
