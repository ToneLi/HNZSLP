
import numpy as np
import json
from lattice_relation import  get_bytes_relations
from collections import  Counter
class Data:

    def __init__(self, data_dir="data/Wiki/", reverse=False):
        self.define_words_len = 4
        self.ngram_len = 15

        self.byte_dic=self.get_lattice_bytes_dic(data_dir)


        self.relation_2_dig_id,self.relation_2_adj_mask_matrix,self.relation_2_com_mask_matrix = self.load_relation_id_mask_matrix(data_dir,self.byte_dic)

        self.train_data = self.load_data(data_dir, "train_triples")


        self.data = self.train_data #+ self.test_data
        self.entities = self.get_entities(self.data)
        self.lattice_relation=np.load(data_dir+"/two_relation_vector.npy")
        # print(self.lattice_relation)


    def get_H_RT(self,data_dir,file):
        path = data_dir + "/" + file
        H_RT_dict=json.load(open(path))

        return H_RT_dict

    def load_data(self, data_dir, data_type="train_triples", reverse=False):


        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split("\t") for i in data]
        return data

    def merge_two_dicts(self,x, y):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = x.copy()
        z.update(y)
        return z

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities


    def get_lattice_bytes_dic(self, data_dir):

        relation_to_describ_dir = data_dir + "/relation_to_surfacename.txt"
        relation_dis_dic = {}
        Ws = []
        with open(relation_to_describ_dir, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line = line.strip().split("\t")
                if line[0] not in relation_dis_dic:
                    space_words=line[1].split(" ")
                    # align_words=self.redefine_words(space_words)
                    for w in space_words:
                        n_gram_word=get_bytes_relations(w,self.ngram_len)
                        # print("----ff",n_gram_word)
                        for tu in n_gram_word:
                            for w in tu:
                                w=w.replace("|"," ").replace("-"," ").split(" ")
                                # print("w",w)

                                for unit in w:
                                # unit=w.split("-")[0].split("|")[1]
                                    Ws.append(unit)
        relation_bytes_set = sorted(list(set(Ws)))
        R_W_ids = {relation_bytes_set[i]: i for i in range(len(relation_bytes_set))}
        R_W_ids["<unk>"]=len(R_W_ids)
        #zation': 2318, 'ze': 2319, 'zen': 2320, '<unk>': 2321}
        return R_W_ids


    def generate_mask_matrix(self, all_words_gram,all_bytes_n):
        adj_points = []
        com_points=[]
        all_bytes = []
        for d in all_words_gram:
            for u in d:
                single_u = u.split("-")
                s = single_u[0].split("|")
                adj_points.append(s[0] + "|" + s[1])
                adj_points.append(s[1] + "|" + s[0])
                adj_points.append(s[1] + "|" + s[2])
                adj_points.append(s[2] + "|" + s[1])
                if len(single_u) == 2:
                    s = single_u[0].split("|")[1]
                    for w in single_u[1].split("|"):
                        com_points.append(s + "|" + w)
                        com_points.append(w + "|" + s)

                unit = u.split("-")[0].split("|")[1]
                all_bytes.append(unit)

        adj_points = set(adj_points)
        com_points=set(com_points)


        adj_mask_matrix = np.zeros([len(all_bytes_n), len(all_bytes_n)])
        com_mask_matrix = np.zeros([len(all_bytes_n), len(all_bytes_n)])


        for i in range(len(all_bytes_n)):
            for j in range(len(all_bytes_n)):
                if all_bytes_n[i] + "|" + all_bytes_n[j] in adj_points:
                    adj_mask_matrix[i, j] = 1
                if all_bytes_n[i] + "|" + all_bytes_n[j] in com_points:
                    com_mask_matrix[i, j] = 1

        return adj_mask_matrix,com_mask_matrix


    def reorg_dig(self,s,ngram_len):
        re_org = []
        for i in range(1, ngram_len):
            single_dig = []
            for dig in s:
                if len(dig) == i:
                    single_dig.append(dig)
            re_org = re_org + single_dig
        return re_org


    def load_relation_id_mask_matrix(self, data_dir,Byte_dic):
        B_dic=Byte_dic

        relation_to_describ_dir = data_dir + "/relation_to_surfacename.txt"
        relation_2_dig_id = {}
        relation_2_adj_mask_matrix = {}
        relation_2_com_mask_matrix = {}

        Ws = []
        FFF=[]
        with open(relation_to_describ_dir, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line = line.strip().split("\t")

                space_words = line[1].split(" ")

                all_bytes=[]

                all_words_gram=[]
                flag = -1
                for w in space_words:  # for each word in align_words, a relation includes many words

                    n_gram_word = get_bytes_relations(w, self.ngram_len)
                    all_words_gram=all_words_gram+n_gram_word
                    for u in n_gram_word:
                        for si in u:
                            flag=flag+1
                            unit=si.split("-")[0].split("|")[1]
                            all_bytes.append(unit)
                all_bytes_n=self.reorg_dig(all_bytes,self.ngram_len)
                # print("all_bytes_n",all_bytes_n)


                if len(all_bytes_n)>70:
                    all_bytes_n=all_bytes_n[:70]
                else:
                    all_bytes_n=all_bytes_n+["<unk>"]*(70-len(all_bytes_n))

                all_bytes_id = [B_dic[i] for i in all_bytes_n]

                adj_mask_matrix,com_mask_matrix=self.generate_mask_matrix(all_words_gram,all_bytes_n)



                if line[0] not in relation_2_dig_id:
                    relation_2_dig_id[line[0]]=all_bytes_id
                if line[0] not in relation_2_adj_mask_matrix:
                    relation_2_adj_mask_matrix[line[0]]=adj_mask_matrix
                if line[0] not in relation_2_com_mask_matrix:
                    relation_2_com_mask_matrix[line[0]]=com_mask_matrix

        return relation_2_dig_id,relation_2_adj_mask_matrix,relation_2_com_mask_matrix
