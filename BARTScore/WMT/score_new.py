import argparse
import os, random
import time
import torch
import numpy as np
from utils import *
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

sys.path.append(str(Path(__file__).absolute().parent.parent))
import sanity_transform

REF_HYPO = read_file_to_list('files/tiny_ref_hypo_prompt.txt')

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger();
logger.handlers = [];
logFormatter = logging.Formatter("%(asctime)s [%(funcName)-15s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def report_stat(s_lis):
    logger.info('upper quartile: %f', np.quantile(s_lis, 0.75))
    logger.info('best: %f avg: %f median: %f std: %f', np.max(s_lis), np.mean(s_lis), np.median(s_lis), np.std(s_lis))


def reduce_mean(args, res_d):  # merge the results of different random seed
    for mn in res_d:
        if args.hypo_transform is not None:
            for hypo_transform in args.hypo_transform.split(','):
                if hypo_transform.endswith('[seed]'):
                    seed_lis = [1, 2, 3, 4, 5]
                    hypo_transform = hypo_transform[:-6]
                    lis_now = []
                    for seed_now in seed_lis:
                        sn = 'ref-A' + '-' + hypo_transform + f'-seed{seed_now}'
                        assert (sn in res_d[mn])
                        lis_now.append(res_d[mn][sn])
                        del res_d[mn][sn]
                    reduce_sn = 'ref-A' + '-' + hypo_transform + '-seedreduce'
                    logger.info('%s reducing randomseed results to %s , lis_now: %s', mn, reduce_sn, str(lis_now))
                    res_d[mn][reduce_sn] = {'mean': np.mean(lis_now)}
    return res_d


def batch_preprocess(lines):
    new_lines = []
    for line in lines:
        line = line.replace("’t", "'t").replace("’s", "'s")
        line = line.replace('“', '"').replace('”', '"')
        new_lines.append(line)
    return new_lines


class Scorer:
    """ Support BLEU, CHRF, BLEURT, PRISM, COMET, BERTScore, BARTScore """

    def __init__(self, args, file_path, device='cuda:0'):
        """ file_path: path to the pickle file
            All the data are normal capitalized, not tokenied, including src, ref, sys
        """
        self.args = args
        self.device = device

        logger.info(f'Data loaded from {file_path}.')
        # if file_path.endswith('.pkl'):
        #    self.data = read_pickle(file_path)
        assert (file_path.endswith('.save'))
        data = torch.load(file_path);
        self.data = data;
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

        self.sys_names = ['ref-A', 'initial', 'sort']  # if you want to run all systems, comment out this line
        transform_d = {'refs': self.refs, 'src': self.srcs}
        if args.hypo_transform is not None:
            for hypo_transform in args.hypo_transform.split(','):
                seed_lis = [1]
                if hypo_transform.endswith('[seed]'):
                    seed_lis = [1, 2, 3, 4, 5]
                    hypo_transform = hypo_transform[:-6]
                for seed_now in seed_lis:
                    tf_lines, tf_stat = sanity_transform.batch_sanity_transform(args, self.data['ref-A'],
                                                                                hypo_transform, transform_d,
                                                                                seed=seed_now)
                    # for kk, hypo in enumerate(self.data['ref-A']):
                    #    sys_lines.append(sanity_transform.sanity_transform(hypo, hypo_transform, src_line = self.srcs[kk], idx = kk, transform_d = transform_d))
                    sn = 'ref-A' + '-' + hypo_transform + f'-seed{seed_now}'
                    logger.info('applied transform as %s edit_ratio(percentage): %f', sn, tf_stat['edit_ratio'] * 100)
                    self.data[sn] = tf_lines
                    self.sys_names.append(sn)

        logger.info('sys_names: %s', str(sys_names))

    def save_data(self, path):
        save_pickle(self.data, path)

    def record(self, scores_better, scores_worse, name):
        """ Record the scores from a metric """
        for doc_id in self.data:
            self.data[doc_id]['better']['scores'][name] = str(scores_better[doc_id])
            self.data[doc_id]['worse']['scores'][name] = str(scores_worse[doc_id])

    def score(self, metrics, sys_lines):
        res_d = {}
        for metric_name in metrics:
            if metric_name == 'bleu':
                from sacrebleu import corpus_bleu
                from sacremoses import MosesTokenizer

                def run_sentence_bleu(candidates: list, references: list) -> list:
                    """ Runs sentence BLEU from Sacrebleu. """
                    tokenizer = MosesTokenizer(lang='en')
                    candidates = [tokenizer.tokenize(mt, return_str=True) for mt in candidates]
                    references = [tokenizer.tokenize(ref, return_str=True) for ref in references]
                    assert len(candidates) == len(references)
                    bleu_scores = []
                    for i in range(len(candidates)):
                        bleus = corpus_bleu([candidates[i], ], [[references[i], ]])
                        bleu_scores.append(bleus.score)
                    return bleu_scores

                start = time.time()
                print(f'Begin calculating BLEU.')
                scores_better = run_sentence_bleu(sys_lines, self.refs)
                print(f'Finished calculating BLEU, time passed {time.time() - start}s.')
                res_d[metric_name] = np.mean(scores_better)

            elif metric_name == 'mover_score':
                from moverscore import word_mover_score, get_idf_dict

                # Set up MoverScore
                with open('files/stopwords.txt', 'r', encoding='utf-8') as f:
                    self.stop_words = set(f.read().strip().split(' '))
                ref_lines = self.refs
                idf_refs = get_idf_dict(ref_lines)

                # if args.hypo_transform is not None: #moved forward
                #    sys_lines = [sanity_transform.sanity_transform(line, args.hypo_transform, transform_d = transform_d, idx = kk) for kk, line in enumerate(sys_lines)] 
                #    sys_lines = [detokenize(ss) for ss in sys_lines]

                idf_hyps = get_idf_dict(
                    sys_lines)  # i calcuate idf here so that it won't be affected the number of system evaluated
                scores = word_mover_score(ref_lines, sys_lines, idf_refs, idf_hyps, self.stop_words,
                                          n_gram=1, remove_subwords=True, batch_size=8, device=self.device)
                res_d[metric_name] = np.mean(scores)

            elif metric_name == 'chrf':
                from sacrebleu import sentence_chrf

                def run_sentence_chrf(candidates: list, references: list) -> list:
                    """ Runs sentence chrF from Sacrebleu. """
                    assert len(candidates) == len(references)
                    chrf_scores = []
                    for i in range(len(candidates)):
                        chrf_scores.append(
                            sentence_chrf(hypothesis=candidates[i], references=[references[i]]).score
                        )
                    return chrf_scores

                start = time.time()
                print(f'Begin calculating CHRF.')
                scores_better = run_sentence_chrf(self.betters, self.refs)
                scores_worse = run_sentence_chrf(self.worses, self.refs)
                print(f'Finished calculating CHRF, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'chrf')

            elif metric_name == 'bleurt':
                from bleurt import score

                def run_bleurt(
                        candidates: list, references: list, checkpoint: str = "models/BLEURT-20",
                        # "models/bleurt-large-512"
                ):
                    scorer = score.BleurtScorer(checkpoint)
                    scores = scorer.score(references=references, candidates=candidates)
                    return scores

                start = time.time()

                scores_better = run_bleurt(sys_lines, self.refs)
                res_d[metric_name] = np.mean(scores_better)
                # self.record(scores_better, scores_worse, 'bleurt')

            elif metric_name == 'prism' or metric_name == 'prismqe':
                # torch.use_deterministic_algorithms(True)
                from prism import Prism

                def run_prism(mt: list, ref: list) -> list:
                    prism = Prism(model_dir="./models/m39v1", lang='en', temperature=1.0)
                    _, _, scores = prism.score(cand=mt, ref=ref, segment_scores=True)
                    return list(scores)

                def run_prism_qe(mt: list, src: list) -> list:
                    prism = Prism(model_dir="./models/m39v1", lang='en', temperature=1.0)
                    scores = prism.score(cand=mt, src=src, segment_scores=True)
                    return list(scores)

                start = time.time()
                if metric_name == 'prism': scores_better = run_prism(sys_lines, self.refs)
                if metric_name == 'prismqe': scores_better = run_prism_qe(sys_lines, self.srcs)
                res_d[metric_name] = np.mean(scores_better)
                # scores_worse = run_prism(self.worses, self.refs)

            elif metric_name == 'comet' or metric_name == 'cometqe':
                # torch.use_deterministic_algorithms(False)
                # from comet import load_from_checkpoint
                args = self.args

                # checkpoint = './models/wmt-large-da-estimator-1718/_ckpt_epoch_1.ckpt'
                # model_path = '/home/gridsan/tianxing/txml_shared/projects/metricnlg_2205/BARTScore/WMT/models/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt'
                # model = load_from_checkpoint(model_path)
                if metric_name == 'comet':
                    model_path = 'models/wmt20-comet-da.save'
                if metric_name == 'cometqe':
                    model_path = 'models/wmt21-comet-qe-mqm.save'  # for this the 'ref' won't be used
                print('loading from', model_path)
                model = torch.load(model_path)

                transform_d = {'refs': self.refs, 'src': self.srcs}
                # logger.info('debug: only do 2000 samples for speed!')

                hyp1_samples = [{'src': self.srcs[i], 'ref': self.refs[i], 'mt': sys_lines[i]} for i in
                                range(len(self.refs))]

                start = time.time()
                scores_better_seg, scores_better = model.predict(hyp1_samples)
                # logger.info('scores for hyp1: %f', scores_better)
                res_d[metric_name] = scores_better
                # self.record(scores_better, scores_worse, 'comet')

            elif metric_name == 'bert_score':
                import bert_score

                def run_bertscore(mt: list, ref: list):
                    """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
                    _, _, f1 = bert_score.score(
                        cands=mt,
                        refs=ref,
                        idf=False,
                        batch_size=32,
                        lang='en',
                        rescale_with_baseline=True,
                        verbose=True,
                        nthreads=4,
                    )
                    return f1.numpy()

                start = time.time()
                # print(f'Begin calculating BERTScore.')
                scores_better = run_bertscore(sys_lines, self.refs)
                res_d[metric_name] = np.mean(scores_better)
                # scores_worse = run_bertscore(self.worses, self.refs)
                # print(f'Finished calculating BERTScore, time passed {time.time() - start}s.')
                # self.record(scores_better, scores_worse, 'bert_score')

            elif metric_name == 'bart_score' or metric_name == 'bart_score_cnn' or metric_name == 'bart_score_para':
                from bart_score import BARTScorer

                def run_bartscore(scorer, mt: list, ref: list):
                    hypo_ref = np.array(scorer.score(mt, ref, batch_size=4))
                    ref_hypo = np.array(scorer.score(ref, mt, batch_size=4))
                    avg_f = 0.5 * (ref_hypo + hypo_ref)
                    return avg_f

                # Set up BARTScore
                if 'cnn' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')

                # start = time.time()
                # print(f'Begin calculating BARTScore.')
                scores_better = run_bartscore(bart_scorer, sys_lines, self.refs)
                res_d[metric_name] = np.mean(scores_better)
                # scores_worse = run_bartscore(bart_scorer, self.worses, self.refs)
                # print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')
                # self.record(scores_better, scores_worse, metric_name)

            elif metric_name.startswith('prompt'):
                """ BARTScore adding prompts """
                from bart_score import BARTScorer

                def prefix_prompt(l, p):
                    new_l = []
                    for x in l:
                        new_l.append(p + ', ' + x)
                    return new_l

                def suffix_prompt(l, p):
                    new_l = []
                    for x in l:
                        new_l.append(x + ' ' + p + ',')
                    return new_l

                if 'cnn' in metric_name:
                    name = 'bart_score_cnn'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    name = 'bart_score_para'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    name = 'bart_score'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')

                start = time.time()
                print(f'BARTScore-P setup finished. Begin calculating BARTScore-P.')
                for prompt in tqdm(REF_HYPO, total=len(REF_HYPO), desc='Calculating prompt.'):
                    ref_better_en = np.array(bart_scorer.score(suffix_prompt(self.refs, prompt), self.betters,
                                                               batch_size=4))
                    better_ref_en = np.array(bart_scorer.score(suffix_prompt(self.betters, prompt), self.refs,
                                                               batch_size=4))

                    better_scores = 0.5 * (ref_better_en + better_ref_en)

                    ref_worse_en = np.array(bart_scorer.score(suffix_prompt(self.refs, prompt), self.worses,
                                                              batch_size=5))
                    worse_ref_en = np.array(bart_scorer.score(suffix_prompt(self.worses, prompt), self.refs,
                                                              batch_size=5))
                    worse_scores = 0.5 * (ref_worse_en + worse_ref_en)
                    self.record(better_scores, worse_scores, f'{name}_en_{prompt}')

                    ref_better_de = np.array(bart_scorer.score(self.refs, prefix_prompt(self.betters, prompt),
                                                               batch_size=5))
                    better_ref_de = np.array(bart_scorer.score(self.betters, prefix_prompt(self.refs, prompt),
                                                               batch_size=5))
                    better_scores = 0.5 * (ref_better_de + better_ref_de)

                    ref_worse_de = np.array(bart_scorer.score(self.refs, prefix_prompt(self.worses, prompt),
                                                              batch_size=5))
                    worse_ref_de = np.array(bart_scorer.score(self.worses, prefix_prompt(self.refs, prompt),
                                                              batch_size=5))
                    worse_scores = 0.5 * (ref_worse_de + worse_ref_de)
                    self.record(better_scores, worse_scores, f'{name}_de_{prompt}')
                print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')

        return res_d


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, default='wmt21/de-en.allsys.new.save',
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--output', type=str, required=False, default=None,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--bleu', action='store_true', default=False,
                        help='Whether to calculate BLEU')
    parser.add_argument('--chrf', action='store_true', default=False,
                        help='Whether to calculate CHRF')
    parser.add_argument('--bleurt', action='store_true', default=False,
                        help='Whether to calculate BLEURT')
    parser.add_argument('--prism', action='store_true', default=False,
                        help='Whether to calculate PRISM')
    parser.add_argument('--prismqe', action='store_true', default=False,
                        help='Whether to calculate PRISM-QE')
    parser.add_argument('--comet', action='store_true', default=False,
                        help='Whether to calculate COMET')
    parser.add_argument('--cometqe', action='store_true', default=False,
                        help='Whether to calculate COMET-QE')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--mover_score', action='store_true', default=False)
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    parser.add_argument('--bart_score_cnn', action='store_true', default=False,
                        help='Whether to calculate BARTScore-CNN')
    parser.add_argument('--bart_score_para', action='store_true', default=False,
                        help='Whether to calculate BARTScore-Para')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Whether to calculate BARTScore-P, can be bart_ref, bart_cnn_ref, bart_para_ref')
    parser.add_argument('--hypo_transform', type=str, default=None,
                        help='transform the hypo (system output) for sanity check purposes')
    parser.add_argument('--debug_transform', action='store_true', default=False)

    args = parser.parse_args()

    scorer = Scorer(args, args.file, args.device)

    METRICS = []
    if args.bleu:
        METRICS.append('bleu')
    if args.chrf:
        METRICS.append('chrf')
    if args.bleurt:
        METRICS.append('bleurt')
    if args.comet:
        METRICS.append('comet')
    if args.cometqe:
        METRICS.append('cometqe')
    if args.bert_score:
        METRICS.append('bert_score')
    if args.mover_score:
        METRICS.append('mover_score')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.bart_score_para:
        METRICS.append('bart_score_para')
    if args.prism: METRICS.append('prism')
    if args.prismqe: METRICS.append('prismqe')
    if args.prompt is not None:
        prompt = args.prompt
        assert prompt in ['bart_ref', 'bart_cnn_ref', 'bart_para_ref']
        METRICS.append(f'prompt_{prompt}')

    res_d = {}
    for sys_name in scorer.sys_names:
        logger.info('Running sys_name: %s', sys_name)
        sys_lines = scorer.data[sys_name]
        res = scorer.score(METRICS, sys_lines)
        for me in res:
            if not me in res_d: res_d[me] = {}
            res_d[me][sys_name] = res[me]

    res_d = reduce_mean(args, res_d)

    for me in res_d:
        logger.info('=== reporting for metric %s ===', me)
        s_lis = []
        refa_score = res_d[me]['ref-A']
        for sn in res_d[me]:
            s_now = res_d[me][sn]['mean'] if isinstance(res_d[me][sn], dict) else res_d[me][sn]
            logger.info('%s: %f refa-percentage: %f', sn, s_now, (s_now - refa_score) / abs(refa_score) * 100.0)
            s_lis.append(res_d[me][sn])
        # report_stat(s_lis)
        logger.info('reporting %s ends here', me)
    # scorer.save_data(args.output)


if __name__ == '__main__':
    main()

"""
python score.py --file kk-en/data.pkl --device cuda:0 --output kk-en/scores.pkl --bleu --chrf --bleurt --prism --comet --bert_score --bart_score --bart_score_cnn --bart_score_para

python score.py --file lt-en/scores.pkl --device cuda:3 --output lt-en/scores.pkl --bart_score --bart_score_cnn --bart_score_para
"""
