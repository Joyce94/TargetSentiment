from data.alphabet import Alphabet
from collections import Counter
import data.utils as utils
import math
import model_att.evaluation as evaluation

class Data():
    def __init__(self):
        self.word_alphabet = Alphabet('word')
        self.category_alphabet = Alphabet('category', is_category=True)
        self.label_alphabet = Alphabet('label', is_label=True)
        self.char_alphabet = Alphabet('char')

        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None

        self.max_char_length = 0

        self.word_num = 0
        self.char_num = 0
        self.label_num = 0


    def build_alphabet(self, word_counter, label_counter, category_counter, shrink_feature_threshold, char=False):
        for word, count in word_counter.most_common():
            if count > shrink_feature_threshold:
                self.word_alphabet.add(word, count)
        for label, count in label_counter.most_common():
            self.label_alphabet.add(label, count)
        for category, count in category_counter.most_common():
            self.category_alphabet.add(category, count)

        ##### check
        if len(self.word_alphabet.word2id) != len(self.word_alphabet.id2word) or len(self.word_alphabet.id2count) != len(self.word_alphabet.id2word):
            print('there are errors in building word alphabet.')
        if len(self.label_alphabet.word2id) != len(self.label_alphabet.id2word) or len(self.label_alphabet.id2count) != len(self.label_alphabet.id2word):
            print('there are errors in building label alphabet.')
        if len(self.category_alphabet.word2id) != len(self.category_alphabet.id2word) or len(self.category_alphabet.id2count) != len(self.category_alphabet.id2word):
            print('there are errors in building category alphabet.')


    def fix_alphabet(self):
        self.word_num = self.word_alphabet.close()
        self.category_num = self.category_alphabet.close()
        self.label_num = self.label_alphabet.close()


    def get_instance(self, file, run_insts, shrink_feature_threshold, char=False, char_padding_symbol='<pad>'):
        words = []
        labels = []
        categorys = []
        insts = []
        word_counter = Counter()
        char_counter = Counter()
        label_counter = Counter()
        category_counter = Counter()

        count = 0
        ner_num = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if run_insts == count: break
                if len(line) > 2:
                    line = line.strip().split(' ')
                    word = line[0]
                    if self.number_normalized: word = utils.normalize_word(word)
                    if len(list(line[1])) > 1:
                        label = line[1].split('-')[0]
                        if label == 'b':
                            ner_num += 1
                        category = line[1].split('-')[1]
                        categorys.append(category)
                        category_counter[category] += 1
                    else:
                        label = line[1]
                    label = label + '-target'
                    words.append(word)
                    labels.append(label)
                    # categorys.append(category)
                    word_counter[word] += 1        #####
                    label_counter[label] += 1
                    # category_counter[category] += 1
                else:
                    insts.append([words, labels, categorys])
                    words = []
                    labels = []
                    categorys = []
                    count += 1
                    if ner_num > 1: print(ner_num)
                    ner_num = 0
        if not self.word_alphabet.fix_flag:
            self.build_alphabet(word_counter, label_counter, category_counter, shrink_feature_threshold, char)
        insts_index = []

        for inst in insts:
            words_index = [self.word_alphabet.get_index(w) for w in inst[0]]
            labels_index = [self.label_alphabet.get_index(l) for l in inst[1]]
            length = len(labels_index)
            categorys_index = [self.category_alphabet.get_index(inst[-1][0])]*length
            # print(len(categorys_index))
            insts_index.append([words_index, labels_index, categorys_index])

        return insts, insts_index


    def build_word_pretrain_emb(self, emb_path, word_dims):
        self.pretrain_word_embedding = utils.load_pretrained_emb_avg(emb_path, self.word_alphabet.word2id, word_dims, self.norm_word_emb)

    def build_char_pretrain_emb(self, emb_path, char_dims):
        self.pretrain_char_embedding = utils.load_pretrained_emb_avg(emb_path, self.char_alphabet.word2id, char_dims, self.norm_char_emb)


    def generate_batch_buckets(self, batch_size, insts, char=False):
        batch_num = int(math.ceil(len(insts) / batch_size))
        buckets = [[[], [], []] for _ in range(batch_num)]
        labels_raw = [[] for _ in range(batch_num)]
        category_raw = [[] for _ in range(batch_num)]
        target_start = [[] for _ in range(batch_num)]
        target_end = [[] for _ in range(batch_num)]

        inst_save = []
        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id == 0 or id % batch_size != 0:
                inst_save.append(inst)
            elif id % batch_size == 0:
                assert len(inst_save) == batch_size
                inst_sorted = utils.sorted_instances_index(inst_save)
                max_length = len(inst_sorted[0][0])
                for idy in range(batch_size):
                    cur_length = len(inst_sorted[idy][0])
                    buckets[idx-1][0].append(inst_sorted[idy][0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][1].append(inst_sorted[idy][1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length))
                    buckets[idx-1][-1].append([1] * cur_length + [0] * (max_length - cur_length))
                    labels_raw[idx-1].append(inst_sorted[idy][1])

                    start, end = evaluation.extract_target(inst_sorted[idy][1], self.label_alphabet)
                    target_start[idx-1].append(start[0])
                    target_end[idx-1].append(end[0])
                    # target_start.extend(start)
                    # target_end.extend(end)
                    category_raw[idx-1].append(inst_sorted[idy][-1][0])
                inst_save = []
                inst_save.append(inst)
        if inst_save != []:
            inst_sorted = utils.sorted_instances_index(inst_save)
            max_length = len(inst_sorted[0][0])
            for idy in range(len(inst_sorted)):
                cur_length = len(inst_sorted[idy][0])
                buckets[batch_num-1][0].append(inst_sorted[idy][0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][1].append(inst_sorted[idy][1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length))
                buckets[batch_num-1][-1].append([1] * cur_length + [0] * (max_length - cur_length))
                labels_raw[batch_num-1].append(inst_sorted[idy][1])
                category_raw[batch_num-1].append(inst_sorted[idy][-1][0])
                start, end = evaluation.extract_target(inst_sorted[idy][1], self.label_alphabet)
                target_start[batch_num - 1].append(start[0])
                target_end[batch_num - 1].append(end[0])
        # print(buckets)
        # print(labels_raw)
        # print(category_raw)
        # print(target_start)
        # print(target_end)
        return buckets, labels_raw, category_raw, target_start, target_end

    def generate_batch_buckets_save(self, batch_size, insts, char=False):
        # insts_length = list(map(lambda t: len(t) + 1, inst[0] for inst in insts))
        # insts_length = list(len(inst[0]+1) for inst in insts)
        # if len(insts) % batch_size == 0:
        #     batch_num = len(insts) // batch_size
        # else:
        #     batch_num = len(insts) // batch_size + 1
        batch_num = int(math.ceil(len(insts) / batch_size))

        if char:
            buckets = [[[], [], [], []] for _ in range(batch_num)]
        else:
            buckets = [[[], [], []] for _ in range(batch_num)]
        max_length = 0
        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id % batch_size == 0:
                max_length = len(inst[0]) + 1
            cur_length = len(inst[0])

            buckets[idx][0].append(inst[0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
            buckets[idx][1].append([self.label_alphabet.word2id['<start>']] + inst[-1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length - 1))
            if char:
                char_length = len(inst[1][0])
                buckets[idx][2].append((inst[1] + [[self.char_alphabet.word2id['<pad>']] * char_length] * (max_length - cur_length)))
            buckets[idx][-1].append([1] * (cur_length + 1) + [0] * (max_length - (cur_length + 1)))

        return buckets











