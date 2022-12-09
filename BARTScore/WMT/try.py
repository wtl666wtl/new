import copy
import torch
import argparse
from numpy import random
import numpy as np
import editdistance

import logging
import bert_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger();
logger.handlers = []
logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

from transformers import (AutoModel, AutoTokenizer, BertModel, RobertaTokenizer, RobertaForMaskedLM)
from transformers import logging

logging.set_verbosity_error()

punctuation = ['.', ':', ',', '/', '?', '<', '>', ';', '[', ']', '{', '}', '-', '_', '`', '~', '+', '=', '\'', '\"',
               '|', '\\']


def batch_preprocess(lines):
    new_lines = []
    for line in lines:
        line = line.replace("’t", "'t").replace("’s", "'s")
        line = line.replace('“', '"').replace('”', '"')
        new_lines.append(line)
    return new_lines


def prepare_sentence(tokenizer, text, id):
    ids = []
    ids.append(tokenizer._convert_token_to_id(tokenizer.bos_token))
    for i in range(len(text)):
        if i == id:
            ids.append(tokenizer._convert_token_to_id(tokenizer.mask_token))
        else:
            for bpe_token in tokenizer.bpe(text[i]).split(" "):
                ids.append(tokenizer._convert_token_to_id(bpe_token))
    ids.append(tokenizer._convert_token_to_id(tokenizer.eos_token))
    return torch.tensor([ids]).long()


class Attacker:
    def __init__(self, args, ratio, file_path, outfile, device='cuda:0'):
        """ file_path: path to the pickle file
            All the data are normal capitalized, not tokenied, including src, ref, sys
        """
        self.args = args
        self.ratio = ratio
        self.device = device
        self.outfile = outfile
        self.filter = args.filter
        self.punc_filter = args.punc_filter

        self.cache = {}

        logger.info(f'Data loaded from {file_path}.')
        # if file_path.endswith('.pkl'):
        #    self.data = read_pickle(file_path)
        assert (file_path.endswith('.save'))
        data = torch.load(file_path)
        self.data = data
        logger.info('preprocessing data...')
        for sn in self.data:
            if sn != 'src':
                self.data[sn] = batch_preprocess(self.data[sn])

        data['src'] = data['src'][:50]
        data['ref-A'] = data['ref-A'][:50]
        data['ref-B'] = data['ref-B'][:50]

        self.srcs = data['src']
        # ref-B is the reference
        self.refs = data['ref-B']
        logger.info("number of texts: %d", len(self.refs))
        sys_names = []
        for sn in self.data:
            if sn != 'ref-B' and sn != 'src':
                sys_names.append(sn)
        self.sys_names = sys_names

        self.srcs = data['src']
        self.sys_names = ['ref-A']  # if you want to run all systems, comment out this line
        transform_d = {'refs': self.refs, 'src': self.srcs}

        logger.info('sys_names: %s', str(sys_names))

        ###################### model and tokenizer ############################
        # MNLI_BERT = 'https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip'
        if args.target == 'bert_score':
            model_type = "./roberta-large"  # default
            self.tokenizer = RobertaTokenizer.from_pretrained(model_type, use_fast=False, do_lower_case=True)
            self.model = RobertaForMaskedLM.from_pretrained(model_type)
            self.model.eval()
            self.model.cuda()
            model = AutoModel.from_pretrained(model_type, local_files_only=True)
            self.embedding = model.embeddings.word_embeddings.weight
        elif args.target == 'bart_score':
            model_type = "facebook/bart-large-cnn"  # default
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, local_files_only=True, use_fast=False)
            self.model = AutoModel.from_pretrained(model_type, local_files_only=True)
            self.model.eval()
            self.embedding = self.model.shared.weight
        elif args.target == 'mover_score':
            import os
            USERHOME = os.path.expanduser("~")
            MOVERSCORE_DIR = os.environ.get('MOVERSCORE', os.path.join(USERHOME, '.moverscore'))
            model_type = os.path.join(MOVERSCORE_DIR)
            self.tokenizer = BertTokenizer.from_pretrained(model_type, use_fast=False)
            self.model = BertModel.from_pretrained(model_type)
            self.model.eval()
            self.embedding = self.model.embeddings.word_embeddings.weight
        elif args.target == 'comet':
            model_type = "./xlm-roberta-large"  # default
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False)
            self.model = AutoModel.from_pretrained(model_type)
            self.model.eval()
            self.embedding = self.model.embeddings.word_embeddings.weight

    def replace(self, w):
        # search
        if w in self.cache:
            return self.cache[w]
        w_id = self.tokenizer._convert_token_to_id(w)
        w_embed = self.embedding[w_id]
        dis = torch.linalg.norm(self.embedding - w_embed, ord=2, axis=1)
        dis, indice = torch.sort(dis)
        index = indice[1].item()
        min_dis = dis[1].item()
        new_w = self.tokenizer._convert_id_to_token(index)
        if self.filter and new_w.lower() == w.lower():
            index = indice[2].item()
            min_dis = dis[2].item()
            new_w = self.tokenizer._convert_id_to_token(index)
        if self.punc_filter and new_w in punctuation and w in punctuation:
            min_dis = 1000000000
        """
        def judge(w):
            for x in w:
                if (x >= '0' and x <= '9') or x in punctuation:
                    return True
            return False
        if self.alpha_filter and judge(w):
            min_dis = 100000000
        rank = 1
        while self.alpha_filter and (new_w.isalpha() == False or new_w[0] == '<'):
            rank += 1
            index = indice[rank].item()
            min_dis = dis[rank].item()
            new_w = self.tokenizer._convert_id_to_token(index)
            if rank > 3 or w.isalpha == False:
                min_dis = 1000000000
                break
        """
        # if self.alpha_filter and w.isalpha() == False:
        #    min_dis = 1000000000
        self.cache[w] = (new_w, min_dis)
        return new_w, min_dis

    def super_replace(self, tokens, id):
        origin = self.tokenizer.convert_tokens_to_string(tokens)
        new_tokens = copy.deepcopy(tokens)
        w = tokens[id]
        w_id = self.tokenizer._convert_token_to_id(w)
        w_embed = self.embedding[w_id]
        dis = torch.linalg.norm(self.embedding - w_embed, ord=2, axis=1)
        dis, indice = torch.sort(dis)
        Q = 8

        def run_bertscore(mt: list, ref):
            """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
            _, _, f1 = bert_score.score(
                cands=mt,
                refs=ref,
                idf=False,
                batch_size=8,
                lang='en',
                rescale_with_baseline=False,
                verbose=False,
                nthreads=8,
            )
            return f1.numpy()

        waiting_list, lines, refs = [], [], []
        for i in range(1, 1 + Q):
            index = indice[i].item()
            min_dis = dis[i].item()
            new_w = self.tokenizer._convert_id_to_token(index)
            if self.filter and new_w.lower() == w.lower():
                index = 0
                min_dis = 1000000000
                new_w = self.tokenizer._convert_id_to_token(index)
            new_tokens[id] = new_w
            new_line = self.tokenizer.convert_tokens_to_string(new_tokens)
            lines.append(new_line)
            refs.append(origin)
            waiting_list.append((new_w, min_dis))

        score = run_bertscore(lines, refs)
        final_index = np.argmax(score)
        new_w, min_dis = waiting_list[final_index]
        return new_w, min_dis

    def random_modify(self, line):
        tokenized_text = self.tokenizer._tokenize(line)
        length = len(tokenized_text)
        # random (maybe we can add some strategies)
        import math
        num = int(math.ceil(self.ratio * length))
        if num == 0:
            return line
        arr = np.array(list(range(length)))
        # arr = random.permutation(arr)
        # search and change

        Q = []
        for i in range(length):
            new_token, dis = self.replace(tokenized_text[i])
            Q.append((dis, i, new_token))
        Q.sort()
        medium_dis, _, _ = Q[length // 2]

        for i in range(num):
            id = arr[i]
            new_token, _ = self.replace(tokenized_text[id])
            if _ >= medium_dis:
                num += 1
                continue
            tokenized_text[id] = new_token
        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def sort_modify2(self, line):
        tokenized_text = self.tokenizer._tokenize(line)
        length = len(tokenized_text)
        import math
        num = int(round(self.ratio * length))
        if num == 0:
            return line
        # sort and change
        Q = []
        for i in range(length):
            new_token, dis = self.replace(tokenized_text[i])
            Q.append((dis, i, new_token))
        Q.sort()
        for i in range(num):
            _, id, new_token = Q[i]
            tokenized_text[id] = new_token
        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def sort_modify3(self, line):
        tokenized_text = self.tokenizer._tokenize(line)
        length = len(tokenized_text)
        import math
        num = int(round(self.ratio * length))
        if num == 0:
            return line
        # sort and change
        Q = []
        for i in range(length):
            new_token, dis = self.super_replace(tokenized_text, i)
            Q.append((dis, i, new_token))
        Q.sort()
        for i in range(num):
            _, id, new_token = Q[i]
            tokenized_text[id] = new_token
        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def sort_modify_(self, line, src):
        import regex as re
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokenized_text = []
        for token in re.findall(self.pat, line):
            token = "".join(
                self.tokenizer.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            #tokenized_text.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
            tokenized_text.append(token)
        #tokenized_text = self.tokenizer._tokenize(line)
        length = len(tokenized_text)
        import math
        num = int(round(self.ratio * length))
        if num == 0:
            return line
        # sort and change
        Q = []

        origin = line
        tokens = tokenized_text

        waiting_list, lines, refs = [], [], []

        def run_bertscore(mt: list, ref):
            """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
            _, _, f1 = bert_score.score(
                cands=mt,
                refs=ref,
                idf=False,
                batch_size=32,
                lang='en',
                rescale_with_baseline=False,
                verbose=False,
                nthreads=8,
            )
            return f1.numpy()

        for id in range(length):
            bpe_check = len(self.tokenizer.bpe(tokens[id]).split(" "))
            if bpe_check > 1:
                continue
            new_tokens = copy.deepcopy(tokens)
            w = tokens[id]
            w_id = self.tokenizer._convert_token_to_id(w)
            w_embed = self.embedding[w_id]
            dis = torch.linalg.norm(self.embedding - w_embed, ord=2, axis=1)
            dis, indice = torch.sort(dis)
            Q = 16

            cnt = 1
            while cnt <= Q:
                cnt += 1
                if Q > 32:
                    break
                index = indice[cnt].item()
                min_dis = dis[cnt].item()
                new_w = self.tokenizer._convert_id_to_token(index)
                if ord(w[0]) == 2**8 + ord(' ') and ord(new_w[0]) != 2**8 + ord(' '):
                    Q += 1
                    continue
                if ord(new_w[0]) == 2**8 + ord(' ') and ord(w[0]) != 2**8 + ord(' '):
                    Q += 1
                    continue
                if self.filter and new_w.lower() == w.lower():
                    Q += 1
                    continue
                if self.punc_filter and w in punctuation:
                    Q += 1
                    continue
                if new_w == self.tokenizer.unk_token or new_w == self.tokenizer.cls_token\
                        or new_w == self.tokenizer.sep_token or new_w == self.tokenizer.pad_token\
                        or new_w == self.tokenizer.mask_token or new_w == self.tokenizer.bos_token\
                        or new_w == self.tokenizer.eos_token:
                    Q += 1
                    continue
                new_tokens[id] = new_w
                new_line = self.tokenizer.convert_tokens_to_string(new_tokens)
                lines.append(new_line)
                refs.append(origin)
                #refs.append(src)
                waiting_list.append((id, new_w, min_dis))

        score = run_bertscore(lines, refs)
        qwq = np.argsort(-score)
        flag = {}
        for i in range(len(qwq)):
            final_index = qwq[i]
            id, new_w, min_dis = waiting_list[final_index]
            if flag.get(id) != 1453:
                flag[id] = 1453
                tokenized_text[id] = new_w
                num -= 1
                if num == 0:
                    break
        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def sort_modify(self, line, src):
        import regex as re
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        tokenized_text = []
        for token in re.findall(self.pat, line):
            token = "".join(
                self.tokenizer.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            #tokenized_text.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
            tokenized_text.append(token)
        #tokenized_text = self.tokenizer._tokenize(line)
        #length = len(tokenized_text)
        import math
        num = int(round(self.ratio * len(tokenized_text)))
        if num == 0:
            return line
        # sort and change
        Q = []

        origin = line
        tokens = tokenized_text

        waiting_list, lines, refs = [], [], []

        def run_bertscore(mt: list, ref):
            """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
            _, _, f1 = bert_score.score(
                cands=mt,
                refs=ref,
                idf=False,
                batch_size=256,
                lang='en',
                rescale_with_baseline=False,
                verbose=False,
                nthreads=8,
            )
            return f1.numpy()

        flag = {}
        while num > 0:
            for id in range(len(tokenized_text)):
                if flag.get(id) == 1453: continue
                bpe_check = len(self.tokenizer.bpe(tokens[id]).split(" "))
                if bpe_check > 1:
                    continue

                new_line = self.tokenizer.convert_tokens_to_string(tokens[:id] + tokens[id + 1:])
                lines.append(new_line)
                refs.append(origin)
                waiting_list.append((id, '<delete>', 0))

                Q = 8

                new_tokens = copy.deepcopy(tokens)
                w = tokens[id]
                ids = prepare_sentence(self.tokenizer, tokenized_text, id)
                with torch.no_grad():
                    output = self.model(ids.cuda())
                predictions = output[0]
                masked_index = (ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
                value, predicted_index = torch.topk(predictions[0, masked_index], k=Q)
                predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in predicted_index]
                for i in range(Q):
                    if self.filter and predicted_token[i].lower() == w.lower():
                        Q += 1
                        continue
                    if self.punc_filter and w in punctuation:
                        Q += 1
                        continue
                    new_tokens[id] = predicted_token[i]
                    new_line = self.tokenizer.convert_tokens_to_string(new_tokens)
                    lines.append(new_line)
                    refs.append(origin)
                    # refs.append(src)
                    waiting_list.append((id, predicted_token[i], 0))

                new_tokens = copy.deepcopy(tokens)
                w = tokens[id]
                w_id = self.tokenizer._convert_token_to_id(w)
                w_embed = self.embedding[w_id]
                dis = torch.linalg.norm(self.embedding - w_embed, ord=2, axis=1)
                dis, indice = torch.sort(dis)

                cnt = 1
                while cnt <= Q:
                    cnt += 1
                    if Q > 10:
                        break
                    index = indice[cnt].item()
                    min_dis = dis[cnt].item()
                    new_w = self.tokenizer._convert_id_to_token(index)
                    if ord(w[0]) == 2 ** 8 + ord(' ') and ord(new_w[0]) != 2 ** 8 + ord(' '):
                        Q += 1
                        continue
                    if ord(new_w[0]) == 2 ** 8 + ord(' ') and ord(w[0]) != 2 ** 8 + ord(' '):
                        Q += 1
                        continue
                    if self.filter and new_w.lower() == w.lower():
                        Q += 1
                        continue
                    if self.punc_filter and w in punctuation:
                        Q += 1
                        continue
                    if new_w == self.tokenizer.unk_token or new_w == self.tokenizer.cls_token \
                            or new_w == self.tokenizer.sep_token or new_w == self.tokenizer.pad_token \
                            or new_w == self.tokenizer.mask_token or new_w == self.tokenizer.bos_token \
                            or new_w == self.tokenizer.eos_token:
                        Q += 1
                        continue
                    new_tokens[id] = new_w
                    new_line = self.tokenizer.convert_tokens_to_string(new_tokens)
                    lines.append(new_line)
                    refs.append(origin)
                    # refs.append(src)
                    waiting_list.append((id, new_w, min_dis))
            score = run_bertscore(lines, refs)
            qwq = np.argsort(-score)
            for i in range(len(qwq)):
                final_index = qwq[i]
                id, new_w, min_dis = waiting_list[final_index]
                if new_w == '<delete>':
                    for rid in range(id, len(tokenized_text)):
                        if flag.get(rid) == 1453:
                            flag[rid] = 0
                            if rid != id:
                                flag[rid-1] = 1453
                    tokenized_text = tokenized_text[:id] + tokenized_text[id+1:]
                    num -= 1
                    break
                if flag.get(id) != 1453:
                    flag[id] = 1453
                    tokenized_text[id] = new_w
                    num -= 1
                    break


        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def work(self, func="sort"):
        self.data[func] = []
        for sys_name in self.sys_names:
            edit_d, ori_len = 0, 0
            logger.info('Running sys_name: %s', sys_name)
            sys_lines = self.data[sys_name]
            flag = 0
            for line, src in zip(sys_lines, self.srcs):
                if func == "sort":
                    new_line = self.sort_modify(line, src)
                else:
                    new_line = self.random_modify(line)
                if self.args.target == "mover_score":  # bad
                    for p in punctuation:
                        new_line = new_line.replace(" " + p, p)
                if flag < 50:
                    flag += 1
                    logger.info('replacing sample %d: \n origin: %s \n %s: %s \n', flag, line, func, new_line)
                self.data[func].append(new_line)

                edit_d += editdistance.eval(new_line.split(), line.split())
                ori_len += len(line.split())
            edit_ratio = edit_d * 1.0 / ori_len
            logger.info('edit_ratio %s', edit_ratio)

    def save_file(self):
        torch.save(self.data, self.outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='wmt21/de-en.allsys.save',
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--output', type=str, default="wmt21/de-en.allsys.new.save",
                        help='The output path to save the calculated scores.')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='The ratio of the tokens in ref-A need to be modified.')
    parser.add_argument('--method', default="sort")
    parser.add_argument('--filter', action='store_true', default=False)
    parser.add_argument('--punc_filter', action='store_true', default=False)
    parser.add_argument('--target', default='bert_score')

    args = parser.parse_args()
    assert 0 <= args.ratio <= 1.0
    assert args.target in ['bert_score', 'bart_score', 'mover_score', 'comet']
    # if args.strong_filter:
    #    assert args.filter, "Please first set filter==True!"

    attack = Attacker(args, args.ratio, args.file, args.output, args.device)
    attack.work(args.method)
    attack.save_file()


if __name__ == '__main__':
    main()
