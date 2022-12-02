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

        key = 2
        data['src'] = [data['src'][key]]
        data['ref-A'] = [data['ref-A'][key]]
        data['ref-B'] = [data['ref-B'][key]]

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

        self.data["sort"] = []
        self.data["sort"].append("An traffic accident occurred on Friday evening on the country road between Machtolsheim and Merklingen where a bike rider was only slightly injured thanks to his quick reaction.")

        logger.info('sys_names: %s', str(sys_names))



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

    args = parser.parse_args()

    attack = Attacker(args, args.ratio, args.file, args.output, args.device)
    attack.save_file()


if __name__ == '__main__':
    main()
