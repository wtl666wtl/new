import copy
import torch
import argparse
from numpy import random
import numpy as np
import editdistance

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(); logger.handlers = []
logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

from transformers import (AutoModel, AutoTokenizer, RobertaTokenizer)

def batch_preprocess(lines):
    new_lines = []
    for line in lines:
        line = line.replace("’t", "'t").replace("’s", "'s")
        line = line.replace('“', '"').replace('”', '"')
        new_lines.append(line)
    return new_lines


class Attacker:
    def __init__(self, args, ratio, file_path, outfile, device='cuda:0'):
        """ file_path: path to the pickle file
            All the data are normal capitalized, not tokenied, including src, ref, sys
        """
        self.args = args
        self.ratio = ratio
        self.device = device
        self.outfile = outfile

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

        self.srcs = data['src']
        # ref-B is the reference
        self.refs = data['ref-B']
        sys_names = []
        for sn in self.data:
            if sn != 'ref-B' and sn != 'src':
                sys_names.append(sn)
        self.sys_names = sys_names

        self.sys_names = ['ref-A']  # if you want to run all systems, comment out this line
        transform_d = {'refs': self.refs, 'src': self.srcs}

        logger.info('sys_names: %s', str(sys_names))

        ###################### model and tokenizer ############################
        #MNLI_BERT = 'https://github.com/AIPHES/emnlp19-moverscore/releases/download/0.6/MNLI_BERT.zip'
        model_type = "roberta-large" # default
        self.tokenizer = AutoTokenizer.from_pretrained("./roberta-large", use_fast=False, do_lower_case=True)
        self.model = AutoModel.from_pretrained("./roberta-large")
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
        self.cache[w] = (new_w, min_dis)
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
        #arr = random.permutation(arr)
        # search and change

        Q = []
        for i in range(length):
            if self.tokenizer.encoder.get(self.tokenizer.unk_token) == tokenized_text[i]:
                continue
            new_token, dis = self.replace(tokenized_text[i])
            Q.append((dis, i, new_token))
        Q.sort()
        medium_dis, _, _ = Q[length // 2]

        min_dis = 1000000000
        for i in range(num):
            id = arr[i]
            if self.tokenizer.encoder.get(self.tokenizer.unk_token) == tokenized_text[id]:
                num += 1
                continue
            new_token, _ = self.replace(tokenized_text[id])
            if _ >= medium_dis:
                continue
            tokenized_text[id] = new_token
        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def initialsentence_modify(self, line):
        tokenized_text = self.tokenizer._tokenize(line)
        length = len(tokenized_text)
        # random (maybe we can add some strategies)
        import math
        num = int(round(self.ratio * length))
        if num == 0:
            return line
        arr = np.array(list(range(length)))
        #arr = random.permutation(arr)
        # search and change
        start = [0]
        for i in range(length):
            if self.tokenizer.encoder.get('.') == tokenized_text[i] and i != length-1:
                start.append(i+1)
        cnt = 0
        tot = 0
        index = 0
        while tot < num:
            tot += 1
            id = start[index] + cnt
            index += 1
            if index == len(start):
                index = 0
                cnt += 1

            if self.tokenizer.encoder.get(self.tokenizer.unk_token) == tokenized_text[id]:
                num += 1
                continue
            new_token, _ = self.replace(tokenized_text[id])
            tokenized_text[id] = new_token

        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def sort_modify(self, line):
        tokenized_text = self.tokenizer._tokenize(line)
        length = len(tokenized_text)
        import math
        num = int(round(self.ratio * length))
        if num == 0:
            return line
        # sort and change
        Q = []
        for i in range(length):
            if self.tokenizer.encoder.get(self.tokenizer.unk_token) == tokenized_text[i]:
                continue
            new_token, dis = self.replace(tokenized_text[i])
            Q.append((dis, i, new_token))
        Q.sort()
        for i in range(num):
            _, id, new_token = Q[i]
            tokenized_text[id] = new_token
        new_line = self.tokenizer.convert_tokens_to_string(tokenized_text)
        return new_line

    def work(self, func="sort"):
        self.data[func] = []
        for sys_name in self.sys_names:
            edit_d, ori_len = 0, 0
            logger.info('Running sys_name: %s', sys_name)
            sys_lines = self.data[sys_name]
            flag = 0
            for line in sys_lines:
                if func == "sort":
                    new_line = self.sort_modify(line)
                elif func == "initialsentence":
                    new_line = self.initialsentence_modify(line)
                else:
                    new_line = self.random_modify(line)
                if flag < 5:
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
                        help='The ratio of the words in ref-A need to be modified.')

    args = parser.parse_args()
    assert 0 <= args.ratio <= 1.0

    attack = Attacker(args, args.ratio, args.file, args.output, args.device)
    attack.work("sort")
    attack.work("initial")
    #attack.work("initialsentence")
    attack.save_file()

if __name__ == '__main__':
    main()
