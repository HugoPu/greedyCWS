# -*- coding: UTF-8 -*-
from collections import defaultdict

import numpy as np
import random
import gensim
import re

def initCemb(ndims,train_file,pre_trained,thr = 5.):
    f = open(train_file)
    train_vocab = defaultdict(float) # {character: number of occurrences}
    for line in f.readlines():
        sent = unicode(line.decode('utf8')).split()
        for word in sent:
            for character in word:
                train_vocab[character]+=1
    f.close()
    character_vecs = {} # {character: vector}
    for character in train_vocab:
        if train_vocab[character]< thr:
            continue
        character_vecs[character] = np.random.uniform(-0.5/ndims,0.5/ndims,ndims)
    # Update pre_trained character vectors character_vecs whose vectors are random
    if pre_trained is not None:
        pre_trained = gensim.models.Word2Vec.load(pre_trained)
        pre_trained_vocab = set([ unicode(w.decode('utf8')) for w in pre_trained.vocab.keys()])
        for character in pre_trained_vocab:
            character_vecs[character] = pre_trained[character.encode('utf8')]
    Cemb = np.zeros(shape=(len(character_vecs)+1,ndims)) # {id:vector}, plus 1 to add <unk> to it
    idx = 1
    character_idx_map = dict() # {character:id}
    for character in character_vecs:
        Cemb[idx] = character_vecs[character]
        character_idx_map[character] = idx
        idx+=1
    return Cemb,character_idx_map

def SMEB(lens):
    idxs = []
    for len in lens:
        for i in xrange(len-1):
            idxs.append(0)
        idxs.append(len)
    return idxs

def prepareData(character_idx_map,path,test=False):
    seqs,wlenss,idxss = [],[],[]
    f = open(path)
    # Loop lines
    for line in f.readlines():
        # Separate line with space
        sent = unicode(line.decode('utf8')).split()
        Left = 0
        for idx,word in enumerate(sent):
            # If this "word" is a punctuation
            if len(re.sub('\W','',word,flags=re.U))==0:
                if idx >Left:
                    # Put this sentence in the sentence list, [sentence[character]]
                    seqs.append(list(''.join(sent[Left:idx])))
                    # Put the length of word in the word length list [sentence[word length]]
                    wlenss.append([len(word) for word in sent[Left:idx]])
                Left = idx+1 # If it is not a punctuation, pointer move right

        # If there isn't any punctuation in the line
        if Left!=len(sent):
            seqs.append(list(''.join(sent[Left:])))
            wlenss.append([ len(word) for word in sent[Left:]])
    # Convert character sequences to index sequences, which is based on the character_idx_map. [sentence[cha_idx]]
    seqs = [[ character_idx_map[character] if character in character_idx_map else 0 for character in seq] for seq in seqs]
    f.close()
    if test:
        return seqs
    # Loop sentence, wlens: word length in one sentence
    for wlens in wlenss:
        idxss.append(SMEB(wlens)) # [sentence[cha_label]]
    return seqs,wlenss,idxss