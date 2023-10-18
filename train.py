import os
import random
import time
import pickle
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import torch
import tqdm
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from my_bert.model_MOVCNet import MOVCNet
from my_bert.optimization import BertAdam, warmup_linear
from my_bert.tokenization import BertTokenizer
from scorer import calculate_metrics
from util import load_ontology, save_result
from data import BufferDataset


def truncate_sentence(sentence, p1, p2, tokenizer, max_length=128):
    s1, s2 = 0, 0
    word_list = sentence.split(' ')
    mention = ' '.join(word_list[p1:p2])
    tokens_all = tokenizer.tokenize(sentence)
    tokens_mention = tokenizer.tokenize(mention)
    limited_length = max_length - 3 - len(tokens_mention)
    if(len(tokens_all)) <= limited_length:
        return_sentence = sentence
    else:
        right_token_num = int((limited_length-len(tokens_mention))/2)
        left_token_num = limited_length - len(tokens_mention) - right_token_num

        left = 0
        for i in range(len(tokens_all)):
            if tokens_mention == tokens_all[i:i+len(tokens_mention)]:
                left = i
        right = left + len(tokens_mention) - 1
        
        if (left-0) >= left_token_num and (len(tokens_all)-right-1) >= right_token_num:
            begin = left - left_token_num
            end = right + right_token_num
            for j in range(begin, begin+5):
                if tokens_all[j][0:2] != '##' and tokens_all[j] != ',':
                    begin = j
            for k in range(end,end-5,-1):
                if tokens_all[k][0:2] != '##' and tokens_all[k] != ',':
                    end = k
            text_return = tokens_all[begin]
            for t in range(begin+1, end+1):
                if tokens_all[t][0:2] != '##':
                    text_return = text_return + ' ' + tokens_all[t]
                else:
                    text_return = text_return + tokens_all[t][2:]
            return_sentence = text_return
        elif (left-0) < left_token_num and (len(tokens_all)-right-1) >= right_token_num:
            begin = 0
            end = limited_length-1
            for j in range(begin, begin+5):
                if tokens_all[j][0:2] != '##' and tokens_all[j] != ',':
                    begin = j
            for k in range(end,end-5,-1):
                if tokens_all[k][0:2] != '##' and tokens_all[k] != ',':
                    end = k
            text_return = tokens_all[begin]
            for t in range(begin+1, end+1):
                if tokens_all[t][0:2] != '##':
                    text_return = text_return + ' ' + tokens_all[t]
                else:
                    text_return = text_return + tokens_all[t][2:]
            return_sentence = text_return
        elif (left-0) >= left_token_num and (len(tokens_all)-right-1) < right_token_num:
            end = len(tokens_all) - 1
            begin = len(tokens_all) - limited_length
            for j in range(begin, begin+5):
                if tokens_all[j][0:2] != '##' and tokens_all[j] != ',':
                    begin = j
            for k in range(end,end-5,-1):
                if tokens_all[k][0:2] != '##' and tokens_all[k] != ',':
                    end = k
            text_return = tokens_all[begin]
            for t in range(begin+1, end+1):
                if tokens_all[t][0:2] != '##':
                    text_return = text_return + ' ' + tokens_all[t]
                else:
                    text_return = text_return + tokens_all[t][2:]
            return_sentence = text_return
        else:
            text_return = 'None'
            return_sentence = text_return
    
    return mention, return_sentence
    

def convert_to_features(mention, sentence, max_seq_length, tokenizer):
    mention_word_list = mention.split(' ')
    tokens_mention = []
    for i, word in enumerate(mention_word_list):
        token = tokenizer.tokenize(word)
        tokens_mention.extend(token)
    textlist = sentence.split(' ')
    tokens = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    if (len(tokens)+len(tokens_mention)) > (max_seq_length-3):
        tokens = tokens[0:(max_seq_length-3-len(tokens_mention))]
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens_mention):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    return input_ids, input_mask, segment_ids


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--train')
    arg_parser.add_argument('--dev')
    arg_parser.add_argument('--test')
    arg_parser.add_argument('--svd')
    arg_parser.add_argument('--ontology')
    arg_parser.add_argument('--output', default='output')
    arg_parser.add_argument('--lr', type=float, default=5e-5)
    arg_parser.add_argument('--max_epoch', type=int, default=100)
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--gpu', default=True)
    arg_parser.add_argument('--device', type=int, default=0)
    arg_parser.add_argument('--buffer_size', type=int, default=200000)
    arg_parser.add_argument('--eval_step', type=int, default=5000)
    
    arg_parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    arg_parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    arg_parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    arg_parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    arg_parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    arg_parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    arg_parser.add_argument('--seed', type=int, default=32, help="random seed for initialization")
    arg_parser.add_argument('--topm', type=int, default=10, help="use top_m obj features")
    args = arg_parser.parse_args()
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # Output directory
    output_dir = os.path.join(args.output, f'{timestamp}_{args.topm}')
    model_mac = os.path.join(output_dir, 'best_mac.mdl')
    model_mic = os.path.join(output_dir, 'best_mic.mdl')
    result_dev = os.path.join(output_dir, 'dev.best.tsv')
    result_test = os.path.join(output_dir, 'test.best.tsv')
    os.mkdir(output_dir)
    
    # Set GPU device
    gpu = torch.cuda.is_available() and args.gpu
    if gpu:
        torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data sets
    print('Loading data sets')
    train_set = BufferDataset(args.train, buffer_size=args.buffer_size)
    dev_set = BufferDataset(args.dev, buffer_size=args.buffer_size)
    test_set = BufferDataset(args.test, buffer_size=args.buffer_size)
    
    # Load obj feature file
    print('----------------Load obj feature file----------------------')
    obj_feat_file = open("dataset/obj_feature.pickle", 'rb')
    obj_feat = pickle.load(obj_feat_file)
    print('----------------Load obj feature file finished-------------')
    
    # Load mention list
    print('----------------Load mention list--------------------------')
    mention_list = []
    with open("dataset/mention.txt") as f:
        for line in f.readlines():
            mention = line.strip()
            mention_list.append(mention)
    print('----------------Load mention list finished-----------------')
    
    # Load file
    all_file = "dataset/all.txt"
    alltxt = pd.read_csv(all_file,sep="\t",names=["p1", "p2", "text", "type", "f"])
    
    # Vocabulary
    label_stoi = load_ontology(args.ontology)
    label_itos = {i: s for s, i in label_stoi.items()}
    label_size = len(label_stoi)
    print('Label size: {}'.format(len(label_stoi)))
    
    # Build BERT model, tokenizer
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = MOVCNet.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=label_size)
    print("bert_model:", args.bert_model)
    if "uncased" in args.bert_model:
        args.do_lower_case = True
    else:
        args.do_lower_case = False

    model.cuda()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        
    eval_step = args.eval_step
    batch_size = args.batch_size
    batch_num = len(train_set) // batch_size
    total_step = args.max_epoch * batch_num
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters, 
                         lr=args.lr, 
                         warmup=args.warmup_proportion,
                         t_total=total_step)
    state = {
        'model': model.state_dict(),
        'args': vars(args),
        'vocab': {'label': label_stoi}
    }
    global_step = 0
    best_scores = {
        'best_acc_dev': 0, 'best_mac_dev': 0, 'best_mic_dev': 0,
        'best_acc_test': 0, 'best_mac_test': 0, 'best_mic_test': 0
    }
    
    for epoch in range(args.max_epoch):
        print('-' * 20, 'Epoch {}'.format(epoch), '-' * 20)
        start_time = time.time()
        epoch_loss = []
        progress = tqdm.tqdm(total=batch_num, mininterval=1,
                         desc='Epoch: {}'.format(epoch))
       
        for batch_idx in range(batch_num):
            model.train()
            global_step += 1
            progress.update(1)
            optimizer.zero_grad()
            batch = train_set.next_batch(label_size, batch_size, drop_last=True, shuffle=True, gpu=gpu)
            (elmos, labels, men_masks, ctx_masks, dists, gathers, men_ids,) = batch
            
            obj_repr = torch.cat([torch.from_numpy(obj_feat[mention_list[men_id]][:args.topm,:]).cuda().unsqueeze(0) for men_id in men_ids], dim=0)
            
            all_input_ids, all_input_mask, all_segment_ids = [], [], []
            for id in men_ids:
                sentence = alltxt['text'][id]
                sentence = sentence.replace('\'\'','"').replace('-LRB-','(').replace('-RRB-',')').replace('-LSB-','[').replace('-RSB-',']').replace('-LCB-','{').replace('-RCB-','}').replace('``','"').replace('/.','.').replace('/?','?')
                men, sen = truncate_sentence(sentence, alltxt['p1'][id], alltxt['p2'][id], tokenizer, args.max_seq_length)
                input_ids, input_mask, segment_ids = convert_to_features(men, sen, args.max_seq_length, tokenizer)
                
                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
            all_input_ids2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_input_ids], dim=0)
            all_input_mask2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_input_mask], dim=0)
            all_segment_ids2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_segment_ids], dim=0)            
            
            neg_log_likelihood = model.forward(all_input_ids2, all_segment_ids2, all_input_mask2, obj_repr, labels)
            neg_log_likelihood.backward()
            optimizer.step()
            epoch_loss.append(neg_log_likelihood.item())
            
            if global_step % eval_step != 0 and global_step != total_step:
                continue
            
            # Dev set
            model.eval()
            best_acc, best_mac, best_mic = False, False, False
            results = defaultdict(list)
            bt_sz = args.batch_size
            for batch in dev_set.all_batches(label_size, batch_size=bt_sz, gpu=gpu):
                elmo_ids, labels, men_masks, ctx_masks, dists, gathers, men_ids = batch
                obj_repr = torch.cat([torch.from_numpy(obj_feat[mention_list[men_id]][:args.topm,:]).cuda().unsqueeze(0) for men_id in men_ids], dim=0)

                all_input_ids, all_input_mask, all_segment_ids = [], [], []
                for id in men_ids:
                    sentence = alltxt['text'][id]
                    sentence = sentence.replace('\'\'','"').replace('-LRB-','(').replace('-RRB-',')').replace('-LSB-','[').replace('-RSB-',']').replace('-LCB-','{').replace('-RCB-','}').replace('``','"').replace('/.','.').replace('/?','?')
                    men, sen = truncate_sentence(sentence, alltxt['p1'][id], alltxt['p2'][id], tokenizer, args.max_seq_length)
                    input_ids, input_mask, segment_ids = convert_to_features(men, sen, args.max_seq_length, tokenizer)
                    
                    all_input_ids.append(input_ids)
                    all_input_mask.append(input_mask)
                    all_segment_ids.append(segment_ids)
                all_input_ids2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_input_ids], dim=0)
                all_input_mask2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_input_mask], dim=0)
                all_segment_ids2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_segment_ids], dim=0)            

                with torch.no_grad():
                    preds = model.predict(all_input_ids2, all_segment_ids2, all_input_mask2, obj_repr)
                
                results['gold'].extend(labels.int().data.tolist())
                results['pred'].extend(preds.int().data.tolist())
                results['ids'].extend(men_ids)
            metrics = calculate_metrics(results['gold'], results['pred'])
            print('---------- Dev set ----------')
            print('Strict accuracy: {:.2f}'.format(metrics.accuracy))
            print('Macro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
                metrics.macro_prec,
                metrics.macro_rec,
                metrics.macro_fscore))
            print('Micro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
                metrics.micro_prec,
                metrics.micro_rec,
                metrics.micro_fscore))
            # Save model
            if metrics.accuracy > best_scores['best_acc_dev']:
                best_acc = True
                best_scores['best_acc_dev'] = metrics.accuracy
            if metrics.macro_fscore > best_scores['best_mac_dev']:
                best_mac = True
                best_scores['best_mac_dev'] = metrics.macro_fscore
                print('Saving new best macro F1 model')
                torch.save(state, model_mac)
                save_result(results, label_itos, result_dev)
            if metrics.micro_fscore > best_scores['best_mic_dev']:
                best_mic = True
                best_scores['best_mic_dev'] = metrics.micro_fscore
                print('Saving new best micro F1 model')
                torch.save(state, model_mic)
                
            # Test set
            results = defaultdict(list)
            bt_sz = args.batch_size
            for batch in test_set.all_batches(label_size, batch_size=bt_sz, gpu=gpu):
                elmo_ids, labels, men_masks, ctx_masks, dists, gathers, men_ids = batch
                obj_repr = torch.cat([torch.from_numpy(obj_feat[mention_list[men_id]][:args.topm,:]).cuda().unsqueeze(0) for men_id in men_ids], dim=0)
                
                all_input_ids, all_input_mask, all_segment_ids = [], [], []
                for id in men_ids:
                    sentence = alltxt['text'][id]
                    sentence = sentence.replace('\'\'','"').replace('-LRB-','(').replace('-RRB-',')').replace('-LSB-','[').replace('-RSB-',']').replace('-LCB-','{').replace('-RCB-','}').replace('``','"').replace('/.','.').replace('/?','?')
                    men, sen = truncate_sentence(sentence, alltxt['p1'][id], alltxt['p2'][id], tokenizer, args.max_seq_length)
                    input_ids, input_mask, segment_ids = convert_to_features(men, sen, args.max_seq_length, tokenizer)
                    
                    all_input_ids.append(input_ids)
                    all_input_mask.append(input_mask)
                    all_segment_ids.append(segment_ids)
                all_input_ids2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_input_ids], dim=0)
                all_input_mask2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_input_mask], dim=0)
                all_segment_ids2 = torch.cat([torch.tensor([x], dtype=torch.long).to(device) for x in all_segment_ids], dim=0)            

                with torch.no_grad():
                    preds = model.predict(all_input_ids2, all_segment_ids2, all_input_mask2, obj_repr)
                
                results['gold'].extend(labels.int().data.tolist())
                results['pred'].extend(preds.int().data.tolist())
                results['ids'].extend(men_ids)
            
            metrics = calculate_metrics(results['gold'], results['pred'])
            print('---------- Test set ----------')
            print('Strict accuracy: {:.2f}'.format(metrics.accuracy))
            print('Macro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
                metrics.macro_prec,
                metrics.macro_rec,
                metrics.macro_fscore))
            print('Micro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
                metrics.micro_prec,
                metrics.micro_rec,
                metrics.micro_fscore))
            if best_acc:
                best_scores['best_acc_test'] = metrics.accuracy
            if best_mac:
                best_scores['best_mac_test'] = metrics.macro_fscore
                save_result(results, label_itos, result_test)
            if best_mic:
                best_scores['best_mic_test'] = metrics.micro_fscore

            for k, v in best_scores.items():
                print('{}: {:.2f}'.format(k.replace('_', ' '), v))
        progress.close()
        

if __name__ == "__main__":
    main()