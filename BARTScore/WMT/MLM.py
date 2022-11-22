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
        #self.filter = args.filter
        #self.punc_filter = args.punc_filter

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
            #self.embedding = self.model.embeddings.word_embeddings.weight
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
        length = len(tokenized_text)
        num = int(round(self.ratio * length))
        if num == 0:
            return line

        arr = np.array(list(range(length)))
        arr = random.permutation(arr)
        # sort and change
        cnt = 0
        while cnt < num:
            if cnt == length:
                break
            id = arr[cnt]
            cnt += 1
            bpe_check = len(self.tokenizer.bpe(tokenized_text[id]).split(" "))
            if bpe_check > 1:
                num += 1
                continue
            ids = prepare_sentence(self.tokenizer, tokenized_text, id)
            with torch.no_grad():
                output = self.model(ids.cuda())
            predictions = output[0]
            masked_index = (ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
            value, predicted_index = torch.topk(predictions[0, masked_index], k=self.args.top_k)
            value = value.cpu().numpy()
            predicted_index = predicted_index.cpu().numpy()
            select_index = np.random.choice(predicted_index, 1, p=value)
            predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in select_index]
            tokenized_text[id] = predicted_token[0]

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
                    new_line = ...
                if self.args.target == "mover_score":  # bad
                    for p in punctuation:
                        new_line = new_line.replace(" " + p, p)
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
                        help='The ratio of the tokens in ref-A need to be modified.')
    parser.add_argument('--method', default="sort")
    #parser.add_argument('--filter', action='store_true', default=False)
    #parser.add_argument('--punc_filter', action='store_true', default=False)
    parser.add_argument('--target', default='bert_score')
    parser.add_argument('--top_k', type=int, default=10)

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
