import re
import logging, random, math
import numpy as np
from nltk import word_tokenize, sent_tokenize
import spacy
import utils

nlp = spacy.load('en_core_web_sm')
logger = logging.getLogger()

def shuffle_word_in_sent(s):
    tts = s.split()
    if len(tts) <= 4:
        return s
    t_h, t_l = tts[0], tts[-1]
    t_body = tts[1:-1]
    random.shuffle(t_body)
    s_shuf = ' '.join([t_h] + t_body + [t_l])
    return s_shuf

#shuffle_word_in_sent('Hi, i am you, while you have a big head, right?')


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
    # end of function lcs

def uppercase_sent_begin(hypo):
    sents = sent_tokenize(hypo); 
    for i in range(len(sents)):
        sents[i] = sents[i][0].upper() + sents[i][1:]
    return ' '.join(sents)

def delete_str_idxs(hypo, idxs):
    hypo_new = ''
    for kk in range(len(hypo)):
        if not kk in idxs:
            hypo_new += hypo[kk]
    return hypo_new

def flu_sentencemiddleswap(hypo, stat_d):
    sents = sent_tokenize(hypo); 
    s_id, max_sl = -1, -1
    for i, sent in enumerate(sents):
        if len(word_tokenize(sent)) > max_sl:
            max_sl = len(word_tokenize(sent))
            s_id = i  
    
    ww = word_tokenize(sents[s_id]); 
    if len(ww) <= 3:
        return hypo, stat_d
    mid_pos = math.ceil(len(ww) * 1.0 / 2)
    if ww[0].lower() in ['the', 'in', 'a', 'on', 'at', 'around', 'when', 'and']:
        ww[0] = ww[0].lower()
    if ww[-1] == '.':
        ww_swap = ww[mid_pos:-1] + ww[0:mid_pos] + [ww[-1]]
    else:
        ww_swap = ww[mid_pos:] + ww[:mid_pos]
    if ww_swap[0] == ',': ww_swap = ww_swap[1:]
    if ww_swap[-2] == ',' and ww_swap[-1] == '.': ww_swap = ww_swap[:-2] + [ww_swap[-1]]
    sent_new = utils.nltk_detokenize(' '.join(ww_swap))
    sent_new = sent_new[0].upper() + sent_new[1:]
    sents[s_id] = sent_new
    return ' '.join(sents), stat_d