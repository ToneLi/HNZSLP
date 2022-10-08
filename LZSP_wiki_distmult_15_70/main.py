from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import json
import random
from tqdm import tqdm


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], data[i][1], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]

        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets



    def evaluate(self, model, mode,entity_sets2ids,renaltion_char_id,lattice_rel):
        test_candidates = json.load(open("data/Wiki/" + mode + "_candidates.json"))

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []
        for query_ in test_candidates.keys():
            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            for e1_rel, tail_candidates in test_candidates[query_].items():
                head, rela = e1_rel.split('\t')

                heads = [entity_sets2ids[head]]
                r_gra=[d.relation_2_dig_id[rela]]
                adj_mask=[d.relation_2_adj_mask_matrix[rela]]

                com_mask=[d.relation_2_com_mask_matrix[rela]]


                candidate_tail_ids = [entity_sets2ids[tail] for tail in tail_candidates]

                e1_idx = torch.tensor(heads)
                if self.cuda:
                    e1_idx = e1_idx.cuda()

                predictions = model.forward(e1_idx, r_gra,adj_mask,com_mask,lattice_rel)
                # print("test_predictions",predictions)
                # unseen_relation_score--tensor([0.6409, 0.4862, 0.6154])
                unseen_relation_score = predictions[0][candidate_tail_ids].cpu()
                # print("unsee",unseen_relation_score)
                sort = list(np.argsort(unseen_relation_score))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)

            print('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(mode + query_, np.mean(hits10_),
                                                                                   np.mean(hits5_), np.mean(hits1_),
                                                                                   np.mean(mrr_)))
            # print('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))

        print('############   ' + mode + '    #############')
        print('HITS10: {:.3f}'.format(np.mean(hits10)))
        print('HITS5: {:.3f}'.format(np.mean(hits5)))
        print('HITS1: {:.3f}'.format(np.mean(hits1)))
        print('MRR: {:.3f}'.format(np.mean(mrr)))
        print('###################################')
        return np.mean(mrr),np.mean(hits10), np.mean(hits5),np.mean(hits1)

    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.r_all_dig_id=d.relation_2_dig_id
        self.adj_mask=d.relation_2_adj_mask_matrix
        self.com_mask=d.relation_2_com_mask_matrix


        train_data_idxs = self.get_data_idxs(d.train_data)

        print("Number of training data points: %d" % len(train_data_idxs))
        #
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)



        er_vocab = self.get_er_vocab(train_data_idxs)
        # print("er_vocab",er_vocab)
        er_vocab_pairs = list(er_vocab.keys())
        fw = open("results.txt", "w", encoding="utf-8")
        print("Starting training...")
        best_score = -float("inf")
        lattice_rel=d.lattice_relation

        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)

            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                da_batch_new=[]
                for da in data_batch:
                    da=list(da)

                    da_n=[da[0]]
                    relation=da[1]

                    da_n.append(self.r_all_dig_id[relation])
                    da_n.append(self.adj_mask[relation])
                    da_n.append(self.com_mask[relation])
                    da_batch_new.append(da_n)

                da_batch_new=np.array(da_batch_new)

                opt.zero_grad()
                e1_idx = torch.tensor([int(x) for x in da_batch_new[:, 0]])
                r_all_ngram_idx = [x for x in da_batch_new[:, 1]]
                r_adj_mask=[x for x in da_batch_new[:, 2]]
                r_com_mask = [x for x in da_batch_new[:, 3]]

                if self.cuda:
                    e1_idx = e1_idx.cuda()

                predictions = model.forward(e1_idx, r_all_ngram_idx,r_adj_mask,r_com_mask,lattice_rel)


                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss = model.loss(predictions, targets)

                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()

            print(" the time of itemizea", it)
            # print("the time assume:",time.time()-start_train)
            print("the meaning loss is:", np.mean(losses))
            model.eval()
            eps = 0.0001
            patience = 5
            with torch.no_grad():
                print("Validation:")
                entity_sets2ids = self.entity_idxs
                renaltion_char_id=d.byte_dic

                best_model = model.state_dict()
                mrr, hit10, hit5, hit1 = self.evaluate(model, "test", entity_sets2ids,renaltion_char_id,lattice_rel)
                fw.write("mrr:" + str(mrr) + "\t" + "hit10:" + str(hit10) + "\t" + "hit5:" + str(
                    hit5) + "\t" + "hit1:" + str(hit1) + "\n")
                fw.flush()
                checkpoint_path = 'wiki_checkpoints/'
                checkpoint_file_name = checkpoint_path + ".pt"

                if hit10 > best_score + eps:
                    no_update = 0
                    torch.save(model.state_dict(), checkpoint_file_name)
                elif (hit10 < best_score + eps) and (no_update < patience):
                    no_update += 1
                elif no_update == patience:
                    torch.save(best_model, checkpoint_path + "best_score_model.pt")
                    exit()
                if it == self.num_iterations - 1:
                    torch.save(best_model, checkpoint_path + "best_score_model.pt")
                    exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Wiki", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=120, nargs="?",  # 500
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=32, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.2, nargs="?",
                        help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.3, nargs="?",
                        help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.3, nargs="?",
                        help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=True)

    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
    experiment.train_and_eval()


