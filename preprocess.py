import os
import itertools
import argparse
import json
import pickle
import random
import spacy
import torch
import numpy as np
import multiprocessing as mp
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import BertTokenizer
from collections import Counter

def save_pickle(d, path):
    with open(path, "wb") as f:
        pickle.dump(d, f)

def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=1)

def bert_encode(tokenizer,
                max_length=20,
                padding='max_length',
                add_special_tokens=True):
    def wrapper(sentence):
        return tokenizer.encode(sentence, 
                                add_special_tokens=add_special_tokens,
                                return_tensors='pt', 
                                max_length=max_length,
                                truncation=True,
                                padding=padding)
    return wrapper

def sentence_to_vec(sentence, nlp):
    vector = np.zeros((1200,))
    tokens = nlp(sentence)
    for i in range(min(3, len(tokens))):
        vector[i*300: (i+1)*300] = tokens[i].vector
        if len(tokens) >= 4:
            for i in range(len(tokens) - 3):
                vector[900:] += tokens[i+3].vector

    return vector

def multi_sentence_to_vec(sentences, nlp):
    vectors = np.zeros((len(sentences), 1200))
    for i, sentence in enumerate(tqdm(sentences)):
        vectors[i, :] = sentence_to_vec(sentence, nlp)
    
    return vectors

def collect_data_by_key(data):
    answers, questions, types, clips = [], [], [], []

    for d in data:
        answers.append(d['a'])
        questions.append(d['q'])
        types.append(d['type'])
        clips.append(d['clip'])
    
    return answers, questions, types, clips

def get_most_popular_answers(answers, top_k):
    from collections import Counter
    answer_count = [(ans, count) for ans, count in Counter(answers).items()]
    answer_count.sort(key=lambda x: x[1], reverse=True)

    answers_unq, _ = zip(*answer_count)
    return list(answers_unq[:top_k]), list(answers_unq[top_k:])

def get_nearest_answers(question_options, question_vector, answer_options, answer, num_nearest_ans):
    distance_vector = np.zeros((question_options.shape[0],))
    for i in range(question_options.shape[0]):
        distance_vector[i] = np.linalg.norm(question_vector-question_options[i, :])
    distances_sort = np.argsort(distance_vector)
    nearest_answers = [answer]

    count_nearest = 0
    for k in range(distances_sort.shape[0]):
        if answer_options[distances_sort[k]] not in nearest_answers:
            nearest_answers.append(answer_options[distances_sort[k]])
            count_nearest += 1
            if count_nearest == num_nearest_ans:
                break
    return nearest_answers[1:]

def generate_options(popular_answers, not_popular_answers, gtruth, nearest_answers):
    options = [[] for _ in range(10)]
    options_class = [[] for _ in range(10)]
    for i in range(len(popular_answers)):
        random_idx = np.random.randint(10)
        while len(options[random_idx]) >= 10:
            random_idx = np.random.randint(10)
        options[random_idx].append(popular_answers[i])
        options_class[random_idx].append('POP')

    if gtruth not in popular_answers:
        random_idx = np.random.randint(10)
        while len(options[random_idx]) >= 10:
            random_idx = np.random.randint(10)
        options[random_idx].append(gtruth)
        options_class[random_idx].append('TRUTH')

    option_itr = 0
    for answer in nearest_answers:
        while len(options[option_itr]) >= 10:
            option_itr += 1
            option_itr = option_itr % 10
        if answer not in popular_answers:
            options[option_itr].append(answer)
            options_class[option_itr].append('NEAR')
            option_itr+=1
            option_itr=option_itr%10

    ope = options

    nonpop_itr=0
    random.shuffle(not_popular_answers)
    for i in range(10):

        while len(options[i]) < 10:
            if (not_popular_answers[nonpop_itr] not in nearest_answers) and (not_popular_answers[nonpop_itr] != gtruth):
                options[i].append(not_popular_answers[nonpop_itr])
                options_class[i].append('RAND')
            nonpop_itr+=1

    options = list(itertools.chain.from_iterable(options))
    options_class = list(itertools.chain.from_iterable(options_class))

    gt_index = options.index(gtruth)
    options_class[gt_index] = 'TRUTH'
    return options, options_class, gt_index

def tokenize_data(data, tokenizer, return_length=False):
    
    def isEmpty(d):
        if isinstance(d, list):
            return not d
        elif isinstance(d, torch.Tensor):
            return d.nelement() == 0

    tokens = []
    tokens_length = []
    for d in tqdm(data):
        tokenized = tokenizer(d)
        if isEmpty(d): 
            continue
        tokens.append(tokenized)
        if return_length:
            tokens_length.append(len(tokenized))
    return tokens if not return_length else tokens, tokens_length

def get_word_count(data):
    word_counts = Counter()
    for d in data:
        word_counts.update(d)
    return word_counts

def encode_vocab(data, word2ind, maxlen):
    assert isinstance(maxlen, int) and maxlen > 0

    unk_token = word2ind['UNK']
    for i, d in enumerate(data):
        data[i] = [word2ind.get(word, unk_token) for word in d][:maxlen] + [0] * (maxlen - len(d))
    
    return data

if __name__ == "__main__":
    random.seed(64)
    np.random.seed(64)
    torch.manual_seed(64)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_json_dir", type=str, default="./data/raw")
    parser.add_argument("-output_dir", type=str, default="./data")
    parser.add_argument("-num_nearest_ans", type=int, default=15)
    parser.add_argument("-top_k_popular_answers", type=int, default=50)
    parser.add_argument('-max_ques_len', default=20, type=int, help="Max length of questions")
    parser.add_argument('-max_ans_len', default=8, type=int, help="Max length of answers")
    parser.add_argument('-word_count_threshold', default=5, type=int, help="Minimum appearance of a word to be included in vocabulary")
    parser.add_argument('-save_vocab', action='store_true', help="Save vocab")
    args = parser.parse_args()

    data_train = json.load(open(os.path.join(args.input_json_dir, "train.json")))
    data_val = json.load(open(os.path.join(args.input_json_dir, "val.json")))
    data_test = json.load(open(os.path.join(args.input_json_dir, "test.json")))

    data = []
    for split in ['train', 'val', 'test']:
        if split == 'train':
            data_split = data_train
        elif split == 'val':
            data_split = data_val
        else: # split == 'test'
            data_split = data_test

        for d in data_split['data']:
            data.append({
                'q': d['q'][:-1].lower(),
                'a': d['a'].lower(),
                'type': d['type'],
                'clip': d['clip'],
                'split': split 
            })
    
    answers, questions, types, clips = collect_data_by_key(data)
    print("Obtaining list of popular answers...")
    answers_pop, answers_nonpop = get_most_popular_answers(answers, args.top_k_popular_answers) 

    unique_answers = list(set(answers)) 
    unique_questions = list(set(questions))

    ##### generate options #####
    nlp = spacy.load('en_core_web_lg')

    print("Creating vectors from question data...")
    questions_vectors = multi_sentence_to_vec(unique_questions, nlp)
    
    def worker(d):
        opt, meta = d
        q = opt['q']
        a = opt['a']
        vector = sentence_to_vec(q, nlp)
        ans_idx = []
        indices = np.random.choice(questions_vectors.shape[0], 4000, replace=False)
        questions_vectors_subset = questions_vectors[indices]
        answers_subset = [answers[i] for i in indices]
        nearest_answers = get_nearest_answers(questions_vectors_subset,
                                            vector,
                                            answers_subset,
                                            a,
                                            args.num_nearest_ans)
        options, option_class, gt_index = generate_options(answers_pop, answers_nonpop, a, nearest_answers)

        for i in range(100):
            ans_idx.append(unique_answers.index(options[i]))
        
        return {'ans_idx': ans_idx, 'gt_index': gt_index}, meta
    

    print('Generating options...')

    opt_list, meta_list = [], []
    for i in range(len(data)):
        opt_list.append({
            'q': data[i]['q'],
            'a': data[i]['a']
        })
        meta_list.append({
            'split': data[i]['split'],
            'clip': data[i]['clip'],
            'type': data[i]['type']
        })
    
    pool = mp.Pool(32)
    opt_data = list(tqdm(pool.imap(worker, zip(opt_list, meta_list)))) # highest overhead
    opt, meta = zip(*opt_data)
    
    pool.close()
    pool.join()

    print("Generate dataset...")

    train_data = []
    val_data = []
    test_data = []

    for i in tqdm(range(len(meta))):
        d = { **opt[i], **meta[i], 'q': data[i]['q'], 'a': data[i]['a'] }
        d['ques_idx'] = unique_questions.index(d['q'])
        if meta[i]['split'] == 'train':
            train_data.append(d)
        elif meta[i]['split'] == 'val':
            val_data.append(d)
        else: # meta[i]['split'] == 'test'
            test_data.append(d)
    
    print("Writing train/val/test data...")

    save_json(train_data, os.path.join(args.output_dir, "train.json"))
    save_json(val_data, os.path.join(args.output_dir, "val.json"))
    save_json(test_data, os.path.join(args.output_dir, "test.json"))
    
    print("Tokenizing data using BertTokenizer...")

    pre_trained_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_tokenizer = bert_encode(pre_trained_tokenizer) 
    tokenized_answers, _ = tokenize_data(unique_answers, bert_tokenizer)
    tokenized_questions, _ = tokenize_data(unique_questions, bert_tokenizer)
    
    save_pickle(tokenized_answers, 
                os.path.join(args.output_dir, "answers_bert.pkl"))
    save_pickle(tokenized_questions, 
                os.path.join(args.output_dir, "questions_bert.pkl"))
    
    print("Tokenizing data using nltk...")

    tokenized_answers, tokenized_answers_length = tokenize_data(unique_answers, 
                                                                word_tokenize,
                                                                return_length=True)
    tokenized_questions, tokenized_questions_length = tokenize_data(unique_questions, 
                                                                    word_tokenize,
                                                                    return_length=True)
    print("Building vocabulary...")
    
    word_counts_total = get_word_count(tokenized_questions) + \
                        get_word_count(tokenized_answers)

    word_counts_total['UNK'] = args.word_count_threshold
    vocab = [word for word, count in word_counts_total.items() \
             if count >= args.word_count_threshold]

    print("Total number of vocabs : {}".format(len(vocab)))
    
    if args.save_vocab:
        print("Saving vocab...")
        with open("vocab.json", "w") as f:
            json.dump(vocab, f)
    
    word2ind = {word: ind for ind, word in enumerate(vocab)}        
    ind2word = {ind: word for word, ind in word2ind.items()}

    save_pickle(tokenized_answers,
                os.path.join(args.output_dir, "answers_rnn_raw.pkl"))
    save_pickle(tokenized_questions,
                os.path.join(args.output_dir, "questions_rnn_raw.pkl"))

    encoded_answers = encode_vocab(tokenized_answers, 
                                word2ind,
                                maxlen=args.max_ans_len)
    encoded_questions = encode_vocab(tokenized_questions, 
                                word2ind,
                                maxlen=args.max_ques_len)    

    save_pickle(encoded_answers,
                os.path.join(args.output_dir, "answers_rnn.pkl"))
    save_pickle(encoded_questions,
                os.path.join(args.output_dir, "questions_rnn.pkl"))
    
    save_pickle(tokenized_answers_length,
                os.path.join(args.output_dir, "answers_rnn_len.pkl"))
    save_pickle(tokenized_questions_length,
                os.path.join(args.output_dir, "questions_rnn_len.pkl"))