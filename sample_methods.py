import os

import editdistance
import numpy as np
import torch
from nltk import PorterStemmer
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn_extra.cluster import KMedoids
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModel

ps = PorterStemmer()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def write_examples(dump_example_path, examples):
    dump_pt = open(file=dump_example_path, mode='w')

    for src, tgt in examples:
        dump_pt.write(src.strip() + '\t' + tgt.strip() + '\n')

    dump_pt.close()

def random_sample(examples, dump_prefix, sample_size = 500):
    dump_path = os.path.join(dump_prefix, "random_sample.txt")
    index_arr = np.arange(len(examples))
    np.random.shuffle(index_arr)
    batch_ids = index_arr[:sample_size]
    sampled_examples = [examples[i] for i in batch_ids]
    write_examples(dump_path, sampled_examples)

    return sampled_examples

def sample_by_perplexity(examples, dump_prefix, sample_size = 500):
    model_id = 'gpt2-medium'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    lls = []
    for i in tqdm(range(len(examples))):
        inputs = tokenizer(examples[i][0], return_tensors='pt', truncation=True,
                                     return_length=True).to(device)

        with torch.no_grad():
            outputs = model(inputs['input_ids'], labels=inputs['input_ids'])
            log_likelihood = outputs.loss.item()
            #print(log_likelihood)
        lls.append(log_likelihood)


    sample_index = [i[0] for i in sorted(enumerate(lls), key=lambda x: x[1])]
    sampled_examples = []

    for top_idx in sample_index[:sample_size]:
        sampled_examples.append(examples[top_idx])
    dump_path = os.path.join(dump_prefix, "sample_by_perplexity_asc.txt")
    write_examples(dump_path, sampled_examples)


    sample_index = [i[0] for i in sorted(enumerate(lls), key=lambda x: x[1], reverse=True)]

    sampled_examples = []

    for top_idx in sample_index[:sample_size]:
        sampled_examples.append(examples[top_idx])
    dump_path = os.path.join(dump_prefix, "sample_by_perplexity_desc.txt")

    write_examples(dump_path, sampled_examples)

def cluster_by_features(examples, dump_prefix, sample_size = 500, feature_type = 'nl_LM_feature'):
    if feature_type == 'nl_LM_feature':
        dump_path = os.path.join(dump_prefix, "cluster_by_LM_features.txt")
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        print("Begin encoding ... ")
        # Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2").to(device)

        sentence_embeddings = []
        for example in tqdm(examples):
            # Tokenize sentences
            encoded_input = tokenizer([example[0]], padding=True, truncation=True, max_length=128, return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling. In this case, mean pooling
            sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            #print(sentence_embedding.size())
            sentence_embeddings.append(sentence_embedding.cpu())
        sentence_embeddings = torch.cat(sentence_embeddings).numpy()
    elif feature_type.endswith('tfidf'):
        if feature_type == 'nl_tfidf':
            dump_path = os.path.join(dump_prefix, "cluster_by_nl_tfidf_features.txt")
            corpus = []
            for src, tgt in examples:
                src_list = src.split(' ')
                stemmed_src = " ".join([ps.stem(token) for token in src_list])
                corpus.append(stemmed_src)

        elif feature_type == 'lf_tfidf':
            dump_path = os.path.join(dump_prefix, "cluster_by_lf_tfidf_features.txt")
            corpus = [example[1] for example in examples]

        pipe = Pipeline([('count', CountVectorizer()), ('tfidf', TfidfTransformer())]).fit(corpus)
        #pipe['count'].transform(corpus).toarray()

        sentence_embeddings = pipe.transform(corpus)
        print(sentence_embeddings.shape)


    print("Begin clustering ... ")
    km = KMeans(n_clusters=sample_size).fit(sentence_embeddings)

    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, sentence_embeddings)

    sampled_examples = []

    for top_idx in closest.tolist():
        sampled_examples.append(examples[top_idx])
    write_examples(dump_path, sampled_examples)

    return sampled_examples

def cluster_by_edit_distance(examples, dump_prefix, sample_size = 500, dist_type = 'nl_edit'):
    if dist_type == 'nl_edit':
        dump_path = os.path.join(dump_prefix, "cluster_by_NL_edit_distance.txt")

    dist_array = np.ndarray((len(examples), len(examples)))

    for out_idx, out_example in tqdm(enumerate(examples)):
        for in_idx, in_example in enumerate(examples):
            if in_idx <= out_idx:
                if dist_type == 'nl_edit':
                    out_src = out_example[0].split(' ')
                    out_src = [ps.stem(token) for token in out_src]
                    in_src = in_example[0].split(' ')
                    in_src = [ps.stem(token) for token in in_src]

                    out_dist = editdistance.eval(out_src, in_src)
                    in_dist = editdistance.eval(in_src, out_src)
                    dist_array[out_idx, in_idx] = (out_dist + in_dist) / 2
                    dist_array[in_idx, out_idx] = (out_dist + in_dist) / 2

    print("kmedoids graph sampling")

    kmedoids = KMedoids(metric='precomputed', n_clusters=sample_size, init='k-medoids++').fit(dist_array)
    # print (kmedoids.medoid_indices_)
    added_inds = kmedoids.medoid_indices_

    added_inds_list = added_inds.squeeze().tolist()

    selected_examples = [examples[indx] for indx in added_inds_list]
    write_examples(dump_path, selected_examples)
    return selected_examples

def read_examples(input_path):
    examples = []
    example_pt = open(file=input_path, mode='r')

    for line in example_pt.readlines():
        line_list = line.split('\t')
        en = line_list[0]
        lf = line_list[1]
        #if len(line_list) > 2:
        #    schema = line_list[2]
        examples.append((en, lf))
    example_pt.close()

    return examples

def sample_examples(sample_size, sample_method, input_path = "example_path.txt",dump_prefix = "sample_results"):
    examples = read_examples(input_path)
    if sample_method == 'random':
        sampled_examples = random_sample(examples, dump_prefix, sample_size = sample_size)
    elif sample_method == 'nl_edit':
        sampled_examples = cluster_by_edit_distance(examples, dump_prefix, sample_size = sample_size, dist_type = 'nl_edit')
    elif sample_method == 'nl_LM_feature':
        sampled_examples = cluster_by_features(examples, dump_prefix, sample_size = sample_size, feature_type = 'nl_LM_feature')
    elif sample_method == 'nl_tfidf':
        sampled_examples = cluster_by_features(examples, dump_prefix, sample_size = sample_size, feature_type = 'nl_tfidf')
    elif sample_method == 'lf_tfidf':
        sampled_examples = cluster_by_features(examples, dump_prefix, sample_size = sample_size, feature_type = 'lf_tfidf')
    elif sample_method == 'perplexity':
        sampled_examples = sample_by_perplexity(examples, dump_prefix, sample_size=sample_size)
    else:
        raise ValueError

    return sampled_examples


def get_param_parser():
    parser = argparse.ArgumentParser(description='AL methods for Robustness Evaluation')
    # paths
    parser.add_argument("--dataset_path", type=str,
                        required=True, help="Path to dataset")
    parser.add_argument("--outputdir", type=str,
                        default='savedir/', help="Output directory")

    parser.add_argument("--sample_size", type=int, default=50)

    parser.add_argument("--sample_method", type=str, default='random',
                        choices=['random', 'nl_edit', 'nl_LM_feature', 'nl_tfidf', 'lf_tfidf', 'perplexity'])
    return parser


def get_params():
    parser = get_param_parser()
    params, unknown = parser.parse_known_args()
    isExist = os.path.exists(params.outputdir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(params.outputdir)
    if len(unknown) > 0:
        raise ValueError("Got unknown parameters " + str(unknown))
    return params

if __name__ == "__main__":
    params = get_params()
    sample_examples(sample_size=params.sample_size, sample_method=params.sample_method, input_path=params.dataset_path, dump_prefix=params.outputdir)

