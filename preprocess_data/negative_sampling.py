"""
The orginal negative_sampling file credits to https://github.com/Guzpenha/transformer_rankers
ns_test dataset, more instructions and explanations added
"""

from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.datasets import preprocess_crr 
from transformers import BertTokenizer
from tqdm import tqdm

import pandas as pd
import argparse
import logging

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run bert ranker for")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output raw negative_samples")
    parser.add_argument("--anserini_folder", default="", type=str, required=False,
                        help="Path containing the anserini bin <anserini_folder>/target/appassembler/bin/IndexCollection")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--seed", default=42, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_ns_train", default=1, type=int, required=False,
                        help="Number of negatively sampled documents to use during training")
    # new args added
    parser.add_argument("--sampler_type", default="random", type=str, required=False,
                        help="Three kinds of sampler can be chosen: random, bm25, sbert")
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    #Load datasets, data file ended with "_l" means label = 1 is added to the orginal MANtis datasets
    add_turn_separator = (args.task != "ubuntu_dstc8") # Ubuntu data has several utterances from same user in the context.
    train = preprocess_crr.read_crr_tsv_as_df(args.data_folder+args.task+"/train_l.tsv", args.sample_data, add_turn_separator)
    valid = preprocess_crr.read_crr_tsv_as_df(args.data_folder+args.task+"/valid_l.tsv", args.sample_data, add_turn_separator)
    test = preprocess_crr.read_crr_tsv_as_df(args.data_folder+args.task+"/test_l.tsv", args.sample_data, add_turn_separator)


    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


    # random negative sampler, args.num_ns_train = 9 
    # For valid and test, because they are relatively small, if 9 negative samples all sampled from themselves, 
    # many samples would repeat a lot. So responses from train are used here.
    if args.sampler_type == "random":    
        ns_train = negative_sampling.RandomNegativeSampler(list(train["response"].values), 1) 
        ns_valid = negative_sampling.RandomNegativeSampler(list(train["response"].values)+list(valid["response"].values), args.num_ns_train)
        ns_test = negative_sampling.RandomNegativeSampler(list(train["response"].values)+list(valid["response"].values), args.num_ns_train)


    # bm25 nagative sampler, args.num_ns_train = 9, anserini_folder needs to be downloaded from here 
    # https://colab.research.google.com/github/castorini/anserini-notebooks/blob/master/anserini_robust04_demo.ipynb
    # which is a notebook from https://github.com/castorini/anserini
    if args.sampler_type == "bm25":
        ns_train = negative_sampling.BM25NegativeSamplerPyserini(list(train["response"].values), 1,
                    args.data_folder+args.task+"/anserini_train/", args.sample_data, args.anserini_folder, set_rm3=True)   
        ns_valid = negative_sampling.BM25NegativeSamplerPyserini(list(train["response"].values)+list(valid["response"].values), args.num_ns_train,
                    args.data_folder+args.task+"/anserini_valid/", args.sample_data, args.anserini_folder, set_rm3=True)
        ns_test = negative_sampling.BM25NegativeSamplerPyserini(list(train["response"].values)+list(test["response"].values), args.num_ns_train,
                    args.data_folder+args.task+"/anserini_test/", args.sample_data, args.anserini_folder, set_rm3=True)
    

    # sentenceBert negative sampler
    if args.sampler_type == "sbert": 
        ns_train = negative_sampling.SentenceBERTNegativeSampler(list(train["response"].values), args.num_ns_train, 
                    args.data_folder+args.task+"/train_sentenceBERTembeds", args.sample_data) 
        ns_valid = negative_sampling.SentenceBERTNegativeSampler(list(train["response"].values)+list(valid["response"].values), args.num_ns_train, 
                    args.data_folder+args.task+"/valid_sentenceBERTembeds", args.sample_data) 
        ns_test = negative_sampling.SentenceBERTNegativeSampler(list(train["response"].values)+list(test["response"].values), args.num_ns_train, 
                    args.data_folder+args.task+"/test_sentenceBERTembeds", args.sample_data) 



    data_dic = {"train":train, "valid":valid, "test":test}
    ns_dic = {"train": ns_train, "valid": ns_valid, "test": ns_test}


    examples_cols = ["context", "relevant_response"] + \
        ["cand_{}_{}".format(args.sampler_type, i) for i in range(args.num_ns_train)] + \
        ["{}_retrieved_relevant".format(args.sampler_type), "{}_rank".format(args.sampler_type)] 
        
    
    for data in ["train", "valid", "test"]:
        examples = []
        logging.info(f"Retrieving candidates using random for {data} dataset")
        for idx, row in enumerate(tqdm(data_dic[data].itertuples(index=False), total=len(data_dic[data]))):
            context = row[0]
            relevant_response = row[1]
            instance = [context, relevant_response]

            ns = ns_dic[data]
            ns_candidates, scores, had_relevant, rank_relevant = ns.sample(context, relevant_response)
            for ns in ns_candidates:
                instance.append(ns)
            instance.append(had_relevant)
            instance.append(rank_relevant)
        examples.append(instance)

        examples_df = pd.DataFrame(examples, columns=examples_cols)
        examples_df.to_csv(args.output_dir+"/{}_{}.tsv".format(data, args.sampler_type), index=False, sep="\t")

if __name__ == "__main__":
    main()