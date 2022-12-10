import re
import logging, random, math
import torch
import numpy as np
from nltk import word_tokenize, sent_tokenize
import spacy, editdistance
import utils
import transform_utils

nlp = spacy.load('en_core_web_sm')
logger = logging.getLogger()

def compute_lcs(line_a, line_b):
    ww_a, ww_b = line_a.split(), line_b.split()
    lcs_num = transform_utils.lcs(ww_a, ww_b)
    return lcs_num

def batch_sanity_transform(args, lines, mode, transform_d, seed = 1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);  
    srcs = transform_d['src']
    tf_lines, stat_d = [], {}
    edit_d, ori_len = 0, 0
    for i, line in enumerate(lines):
        tf_line, sd_now = sanity_transform(args, line, mode, src_line = srcs[i], idx = i, transform_d = transform_d)
        tf_lines.append(tf_line)
        edit_d += editdistance.eval(tf_line.split(), line.split()); ori_len += len(line.split())
    
    stat_d['edit_ratio'] = edit_d * 1.0 / ori_len
    
    return tf_lines, stat_d

def sanity_transform(args, hypo, mode, src_line = None, idx = None, transform_d = None):
    stat_d = {}

    if mode.startswith('randomworddrop-'):
        prob = float(mode.split('-')[-1])
        ww = hypo.split()
        drop_num = math.ceil(len(ww) * prob)
        if drop_num >= len(ww):
            drop_num = max(len(ww) - 1, 0)
        for kk in range(drop_num):
            drop_idx = random.randint(0, len(ww) - 1)
            ww = ww[:drop_idx] + ww[drop_idx + 1:]
        hypo_new = ' '.join(ww)
    
    if mode.startswith('lastwords-'):
        #lastwords-4-rep-2
        rep_num = int(mode.split('-')[-1])
        word_num = int(mode.split('-')[1])
        #if hypo[-1] == '.': #qualitatively, i found that deleting period gives less log-prob for bart_cnn
        #    hypo = hypo[:-1] #delete the last period
        rep_w = ' '.join(hypo.split(' ')[-word_num:])
        if rep_w.endswith('.'):
            rep_w = rep_w[0].upper() + rep_w[1:]
        hypo_new = hypo + (' ' + rep_w) * rep_num

    if mode.startswith('rep-repwhole-'):
        rep_num = int(mode.split('-')[-1])
        hypo_new = ' '.join([hypo] * (rep_num + 1))

    if mode.startswith('rep-lastsenrep-'):
        rep_num = int(mode.split('-')[-1])
        sens = sent_tokenize(hypo)
        last_s = sens[-1] 
        hypo_new = ' '.join([hypo] + [last_s] * rep_num)

    if mode.startswith('flu-removestopwords-'):
        prob = float(mode.split('-')[-1])
        doc = nlp(hypo); 
        delete_idx = []
        for token in doc:
            if token.is_stop == True:
                if random.random() <= prob:
                    #logger.info(token.text + ' ' + hypo)
                    if token.idx > 0 and hypo[token.idx - 1] == ' ':
                        delete_idx.append(token.idx - 1)    
                    for kk in range(token.idx, token.idx + len(token.text)):
                        delete_idx.append(kk)

        hypo_new = transform_utils.delete_str_idxs(hypo, delete_idx)
        hypo_new = hypo_new.replace('  ', ' ').strip()
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)        

    if mode.startswith('flu-truncate-'):
        ww = hypo.split()
        trunc_num = math.floor(len(ww) * float(mode.split('-')[-1]))
        ww = ww[:len(ww) - trunc_num]
        assert(len(ww) >= 1)
        hypo_new = ' '.join(ww)
        if hypo_new[-1] != '.': hypo_new += '.'

    if mode.startswith('flu-lemmatizeverb'):
        doc = nlp(hypo); hypo_new = hypo;
        for token in doc:
            if token.pos_ in ['VERB']: #'VERB':
                hypo_new = hypo_new.replace(token.text, token.lemma_)
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)

    if mode.startswith('flu-removearticle'):
        ww_new = []
        for w in hypo.split():
            if not w.lower() in ['a', 'an', 'the']: ww_new.append(w)
        hypo_new = ' '.join(ww_new)
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)
    
    if mode.startswith('flu-removepreposition-'):
        #prep_lis = ['for', 'in', 'on', 'with', 'by', 'inside', 'outside']
        doc = nlp(hypo); hypo_new = hypo;
        remove_prob = float(mode.split('-')[-1])
        delete_idx = []
        for token in doc:
            if token.pos_ in ['ADP']: #'VERB':
                #logger.info(token.text + ' ' + token.pos_ + ' ' + hypo)
                if random.random() < remove_prob:
                    for kk in range(token.idx, token.idx + len(token.text)):
                        delete_idx.append(kk)
                    #hypo_new = hypo_new.replace(token.text, '', 1) #only replace one occurance
        hypo_new = transform_utils.delete_str_idxs(hypo, delete_idx)
        hypo_new = hypo_new.replace('  ', ' ').strip()
        hypo_new = transform_utils.uppercase_sent_begin(hypo_new)    

    if mode.startswith('flu-removepunct'):
        hypo_new = hypo.replace(',', '')
        hypo_new = hypo_new.replace('"', '')
        hypo_new = hypo_new.replace('.', '')
        hypo_new = hypo_new.replace(':', '')

    if mode.startswith('flu-noisepunct'):
        hypo_new = ''
        for i in range(len(hypo)):
            if hypo[i] == '.':
                hypo_new += ','
            elif hypo[i] == ',':
                hypo_new += '.'
            elif hypo[i] in ['!', '?']:
                hypo_new += ','
            elif hypo[i] == ':':
                hypo_new += ''
            else:
                hypo_new += hypo[i]

    if mode.startswith('flu-sentencemiddleswap'):
        hypo_new, stat_d = transform_utils.flu_sentencemiddleswap(hypo, stat_d)

    if mode.startswith('flu-randomlocalswap-'):
        prob = float(mode.split('-')[-1])
        ww = hypo.split()
        s_num = math.ceil(len(ww) * prob)
        for kk in range(s_num):
            s_idx = random.randint(0, len(ww) - 2) #we will swap the token and the token after it
            ww = ww[:s_idx] + [ww[s_idx + 1]] + [ww[s_idx]] + ww[s_idx + 2:]
        hypo_new = ' '.join(ww)

    if mode.startswith('flu-randomtokenrep-'):
        prob = float(mode.split('-')[-1])
        ww = hypo.split()
        s_num = math.ceil(len(ww) * prob)
        for kk in range(s_num):
            s_idx = random.randint(0, len(ww) - 1) 
            ww = ww[:s_idx] + [ww[s_idx]] + [ww[s_idx]] + ww[s_idx + 1:]
        hypo_new = ' '.join(ww)

    if mode.startswith('flu-shufflewordinsent'):
        sens = sent_tokenize(hypo)
        s_id = random.randint(0, len(sens) - 1)
        sens[s_id] = transform_utils.shuffle_word_in_sent(sens[s_id])
        hypo_new = ' '.join(sens)

    if mode.startswith('alllower'):
        hypo_new = hypo.lower()

    if mode.startswith('addlastperiod'):
        hypo_new = hypo + '.'

    if mode.startswith('removelastperiod'):
        if not any([hypo.endswith(ww) for ww in ['.', '?', '!', '"', ':']]):
            logger.info('meet hypo not ending with period: %s', hypo)
            hypo_new = hypo
        else:
            hypo_new = hypo[:-1]

    if mode.startswith('refner-'):
        ref = transform_d['refs'][idx]
        ty = mode.split('-')[1]
        assert(ty == 'person')
        ref = ref.replace('Sir ', '').replace('Mr ', '').replace('Mr. ', '')
        while 1 == 1: #every time we modify the string, the index is changed, so we do "nlp" again
            doc = nlp(ref)
            found_new = False
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    replace_w = 'he'
                    pref = ref[max(ent.start_char - 11, 0):ent.start_char]
                    if any([(ww in pref) for ww in [' to ', ' with ', ' from ', ' over ', ' by ', ' beat ', ' defeat ', ' defeated ', ' shot ', ' down ']]):
                        replace_w = 'him'
                    if any([(ww in pref) for ww in [' the ']]):
                        replace_w = 'man'
                    if any([pref.endswith(ww) for ww in [' boss ', ' reporter ', ' no 1 ', ' striker ']]):
                        replace_w = ''
                    #if any([pref.lower().endswith(ww) for ww in ['sir ', 'mr ', 'mr. ']]):
                    #    replace_w = ent.text
                    if ent.text.endswith("'s"):
                        replace_w = 'his'
                    if pref.endswith('. ') or (ent.start_char == 0):
                        replace_w = replace_w[0].upper() + replace_w[1:]
                    ref_new = ref[:ent.start_char] + replace_w + ref[ent.end_char:]
                    #print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    #breakpoint()
                    ref = ref_new
                    found_new = True
                    break
            if found_new is False:
                break

        hypo_new = ref

    if mode.startswith('refspanrep-'):
        rep_num = int(mode.split('-')[1])
        #sen = sent_tokenize(src_line)[0]
        sen = transform_d['refs'][idx]
        #if sen[-1] == '.': sen = sen[:-1]
        rep_w = ' '.join(sen.split(' ')[-3:])
        if rep_w.endswith('.'):  #looks like both retaining the period and uppcase is useful!
            rep_w = rep_w[0].upper() + rep_w[1:]
        sen_rep = sen + (' ' + rep_w) * rep_num
        hypo_new = sen_rep

    if mode.startswith('srcsenrep-'):
        rep_num = int(mode.split('-')[1])
        sens = sent_tokenize(src_line)
        hypo_new = ' '.join([sens[0]] * rep_num)

    if mode.startswith('highfreqsource-'):
        top_num = int(mode.split('-')[1][3:])
        len_num = int(mode.split('-')[2][3:])
        src_tt = word_tokenize(src_line)
        freq_d = {}
        N_GRAM = 2
        for i in range(len(src_tt) - N_GRAM + 1):
            w = ' '.join(src_tt[i: i + N_GRAM])
            if not w in freq_d: freq_d[w] = 0
            freq_d[w] = freq_d[w] + 1
        w_d_sorted = sorted(freq_d.items(), key = lambda x: x[1], reverse = True)
        freq_w = [w[0] for w in w_d_sorted[:top_num]]
        gen_lis = []
        for i in range(len_num):
            gen_lis.append(freq_w[random.randint(0, top_num - 1)])
        hypo_new = ' '.join(gen_lis)

    if mode.startswith('highfreqrandom-'):
        top_num = int(mode.split('-')[1][3:])
        len_num = int(mode.split('-')[2][3:])
        w_d = transform_d['wfreq_d']
        w_d['.'], w_d[','] = -1000, -1000 #make sure they do not appear
        w_d_sorted = sorted(w_d.items(), key = lambda x: x[1], reverse = True)
        freq_w = [w[0] for w in w_d_sorted[:top_num]]
        gen_lis = []
        for i in range(len_num):
            gen_lis.append(freq_w[random.randint(0, top_num - 1)])
        
        #refs = transform_d['refs']
        hypo_new = ' '.join(gen_lis)

    if mode.startswith('refreservesort-'):
        sort_m, ref_id = mode.split('-')[-2], int(mode.split('-')[-1])
        if sort_m == 'freq': w_d = transform_d['wfreq_d']
        if sort_m == 'logprob': w_d = transform_d['wlogprob_d']
        ref_reserve = transform_d['refs_reserve'][idx]
        ref_scores = []
        for ref in ref_reserve:
            ref_tt = word_tokenize(ref.lower())
            score = []
            for w in ref_tt:
                if not w in w_d:
                    logger.info('warning: word [%s] not in w_d, skipping...', w)
                    continue
                score.append(w_d[w])
            score = np.mean(score)
            ref_scores.append((ref, score))
        ref_scores = sorted(ref_scores, key = lambda x: x[1], reverse = True)
        hypo_new = ref_scores[ref_id][0]
    
    if mode.startswith('refreserve-'):
        ref_id = int(mode.split('-')[-1])
        ref_reserve = transform_d['refs_reserve']
        hypo_new = ref_reserve[idx][ref_id]

    if mode.startswith('longestrefreserve-'):
        ref_id = int(mode.split('-')[-1])
        refs = transform_d['refs_reserve'][idx]
        refs = [(len(ww.split()), ww) for ww in refs]
        sorted_ref = sorted(refs, key = lambda x: x[0], reverse = True)
        hypo_new = sorted_ref[ref_id][1]

    if mode.startswith('useref'):
        refs = transform_d['refs']
        hypo_new = refs[idx]
        
    if mode.startswith('modelgen'):
        model_gens = transform_d['model_gens']
        hypo_new = model_gens[idx]

    if mode.startswith('copysrc'):
        hypo_new = src_line
     
    if mode.startswith('deletelastsen-'):
        del_num = int(mode.split('-')[-1])
        pos = [_.start() for _ in re.finditer('\.', hypo)] 
        if len(pos) <= 1:
            return hypo
        if len(pos) <= del_num:
            del_num = len(pos) - 1
        hypo_new = hypo[:pos[- del_num - 1] + 1]
 
    if mode.startswith('switchsentence'):
        pos = [_.start() for _ in re.finditer('\.', hypo)] 
        if len(pos) <= 1:
            return hypo
        pos = [-1] + pos
        if pos[-1] != len(hypo) - 1: pos = pos + [len(hypo) - 1]
        sens = [hypo[pos[l - 1] + 1: pos[l] + 1] for l in range(1, len(pos))]
        assert(''.join(sens) == hypo)
        if sens[1].startswith(' ') and not sens[0].startswith(' '):
            sens[0] = ' ' + sens[0]
        
        sp = random.randint(1, len(sens) - 1)
        sens[sp - 1], sens[sp] = sens[sp], sens[sp - 1]
        
        hypo_s = ''.join(sens)
        if hypo_s.startswith(' '): hypo_s = hypo_s[1:]
        hypo_new = hypo_s
        
    if mode.startswith('delduptoken'):
        dd_hypo = []
        for tt in hypo.split(' '):
            #if (tt in [',', '.']) or (not (tt in dd_hypo and tt in ['a', 'the', 'an', 'and'])):
            if (tt in [',', '.']) or (not (tt in dd_hypo)):
                dd_hypo.append(tt)
            #else: print(tt, ' ')
        hypo_new = ' '.join(dd_hypo)

    if mode.startswith('article'):
        words = ['a', 'an', 'A', 'An']
        replace = {}
        for i in range(len(words)):
            replace[words[i]] = words[i ^ 1]
        dd_hypo = []
        for tt in hypo.split(' '):
            #if (tt in [',', '.']) or (not (tt in dd_hypo and tt in ['a', 'the', 'an', 'and'])):
            if tt in words:
                dd_hypo.append(replace[tt])
            else:
                dd_hypo.append(tt)
            #else: print(tt, ' ')
        hypo_new = ' '.join(dd_hypo)

    if mode.startswith('this_these'):
        words = ['this', 'these', 'This', 'These']
        replace = {}
        for i in range(len(words)):
            replace[words[i]] = words[i ^ 1]
        dd_hypo = []
        for tt in hypo.split(' '):
            #if (tt in [',', '.']) or (not (tt in dd_hypo and tt in ['a', 'the', 'an', 'and'])):
            if tt in words:
                dd_hypo.append(replace[tt])
            else:
                dd_hypo.append(tt)
            #else: print(tt, ' ')
        hypo_new = ' '.join(dd_hypo)

    if mode.startswith('number'):
        words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen']
        words_2 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen']
        replace = {}
        for i in range(len(words)-1):
            replace[words[i]] = words[i + 1]
        for i in range(len(words_2)-1):
            replace[words[i]] = words[i + 1]
        dd_hypo = []
        for tt in hypo.split(' '):
            #if (tt in [',', '.']) or (not (tt in dd_hypo and tt in ['a', 'the', 'an', 'and'])):
            if tt in words:
                dd_hypo.append(replace[tt])
            else:
                dd_hypo.append(tt)
            #else: print(tt, ' ')
        hypo_new = ' '.join(dd_hypo)

    if mode.startswith('weekday'):
        words = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        replace = {}
        for i in range(len(words) - 1):
            replace[words[i]] = words[i + 1]
        replace['Friday'] = 'Thursday'
        replace['Sunday'] = 'Saterday'
        replace['Saterday'] = 'Sunday'
        dd_hypo = []
        for tt in hypo.split(' '):
            if tt in words:
                dd_hypo.append(replace[tt])
            else:
                dd_hypo.append(tt)
        hypo_new = ' '.join(dd_hypo)

    if mode.startswith('pronoun'):
        words = ['he', 'she', 'it', 'they']
        words2 = ['He', 'She', 'It', 'They']
        dd_hypo = []
        for tt in hypo.split(' '):
            if tt in words:
                dd_hypo.append(words[random.randint(0,3)])
            elif tt in words2:
                dd_hypo.append(words2[random.randint(0,3)])
            else:
                dd_hypo.append(tt)
        hypo_new = ' '.join(dd_hypo)

    if args.debug_transform and idx <= 20:
        logger.info('idx: %d, hypo    : %s', idx, hypo)
        logger.info('idx: %d, hypo_new: %s', idx, hypo_new)
        logger.info('idx: %d, ref     : %s', idx, transform_d['refs'][idx])

    return hypo_new, stat_d