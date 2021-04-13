""" Module for handling data files """

import gzip
import os
import sys
import torch
import numpy as np
import dill
<<<<<<< HEAD
=======
import re
>>>>>>> sim

re_sentend = re.compile(r'(?<!\b[A-Z]\.)(?<!\b[Mm]rs\.)(?<!\b[MmDdSsJj]r\.)(?<=[\.\?\!])[ \n\t](?!["\'])|(?<!\b[A-Z]\.)(?<!\b[Mm]rs\.)(?<!\b[MmDdSsJj]r\.)(?<=[\.\?\!] ["\'])[ \n\t]+')

def sent_tokenize(instr):
    return(re.split(re_sentend,instr))

def match_embeddings(idx2w, w2vec, dim, bigram=False):
    embeddings = []
    voc_size = len(idx2w)
    print("Matching embeddings to vocabulary ids...")

    for idx in range(voc_size):
        word = idx2w[idx]

        #map unk and eos appropriately
        if word == "<unk>":
            word = 'unk'
        if word == '<eos>':
            word = 'eos'

        #if word not in embeddings make random
        if word not in w2vec:
            embeddings.append(np.random.uniform(low=-1.2, high=1.2, size=(dim,)))
        else:
            embeddings.append(w2vec[word])

    embeddings = np.stack(embeddings)
    return embeddings

def match_embeddings(idx2w, w2vec, dim, bigram=False):
    embeddings = []
    voc_size = len(idx2w)
    print("Matching embeddings to vocabulary ids...")

    for idx in range(voc_size):
        word = idx2w[idx]

        #map unk and eos appropriately
        if word == "<unk>":
            word = 'unk'
        if word == '<eos>':
            word = 'eos'

        #if word not in embeddings make random
        if word not in w2vec:
            embeddings.append(np.random.uniform(low=-1.2, high=1.2, size=(dim,)))
        else:
            embeddings.append(w2vec[word])

    embeddings = np.stack(embeddings)
    return embeddings

def isfloat(instr):
    """ Reports whether a string is floatable """
    try:
        _ = float(instr)
        return(True)
    except:
        return(False)

class Dictionary(object):
    """ Maps between observations and indices """
    def __init__(self, embeddingfname=None, fasttext_loc=None, allowOOV=False):
        self.word2idx = {}
        self.idx2word = []
        #For words that don't have pretrained embeddings
        self.allowOOV = allowOOV
        if embeddingfname:
            self.w2vec = self.load_embeddings(embeddingfname, fasttext_loc)
        else:
            self.w2vec = None
        self.embeddings = []

    def load_embeddings(self, w2vec_loc, fasttext_loc):

        sys.stderr.write('Loading pretrained embeddings...\n')
        if os.path.exists(w2vec_loc):
            with open(w2vec_loc, 'rb') as f:
                w2vec = dill.load(f)
        else:
            w2vec = {}
            with open(fasttext_loc) as f:
                f.__next__()
                for line in f:
                    items = line.strip().split(' ')
                    token = items[0]

                    #map unk and eos appropriately
                    if token == 'unk':
                        token = '<unk>'
                    if token == 'eos':
                        token = '<eos>'
<<<<<<< HEAD

                    vector = np.array(items[1:]).astype(float)
                    w2vec[token] = vector
            with open(w2vec_loc, 'wb') as f:
                dill.dump(w2vec, f)
        
        return w2vec

    def match_embeddings(self):
        voc_size = len(self.idx2word)
        embed_dim = len(list(self.w2vec.values())[0])
        sys.stderr.write("Matching embeddings to vocabulary ids...\n")

        embeddings = []

=======

                    vector = np.array(items[1:]).astype(float)
                    w2vec[token] = vector
            with open(w2vec_loc, 'wb') as f:
                dill.dump(w2vec, f)
        
        return w2vec

    def match_embeddings(self):
        voc_size = len(self.idx2word)
        embed_dim = len(list(self.w2vec.values())[0])
        sys.stderr.write("Matching embeddings to vocabulary ids...\n")

        embeddings = []

>>>>>>> sim
        for idx in range(voc_size):
            word = self.idx2word[idx]
            if self.allowOOV and word not in self.w2vec:
                embeddings.append(np.random.uniform(low=-1.2, high=1.2, size=(embed_dim,)))
            else:
                try:
                    embeddings.append(self.w2vec[word])
                except KeyError:
                    sys.stderr.write('You do not have an'+ \
                            ' embedding for '+word+'. This '+\
                            'will cause problems in training...'+\
                            'aborting\n')
                    sys.exit(1)
        self.embeddings = np.stack(embeddings)

    def add_word(self, word, maxVocab=50000):
        """ Adds a new obs to the dictionary if needed """

        #if i can't make up embeddings and I'm using embeddings I need to 
        #be careful to add only words which I have embeddings for
        if not self.allowOOV and self.w2vec is not None:
            if word not in self.w2vec:
                word = '<unk>'

        if word not in self.word2idx:

            if len(self.idx2word) > maxVocab:
                word = '<unk>'

            else:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class SentenceCorpus(object):
    """ Loads train/dev/test corpora and dictionary """
    def __init__(self, path, vocab_file, test_flag=False, interact_flag=False,
                 checkpoint_flag=False, predefined_vocab_flag=False, lower_flag=False,
                 collapse_nums_flag=False,multisentence_test_flag=False,generate_flag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt', 
                 embeddingfname = None, 
                 fasttext_loc = None, 
                 allowOOV=False):
        self.lower = lower_flag
        self.collapse_nums = collapse_nums_flag
        if not (test_flag or interact_flag or checkpoint_flag or predefined_vocab_flag or generate_flag):
            # training mode
            self.dictionary = Dictionary(embeddingfname, fasttext_loc, allowOOV)
            self.train = self.tokenize(os.path.join(path, trainfname))
            print('after train the len is', len(self.dictionary))
            self.valid = self.tokenize_with_unks(os.path.join(path, validfname))
            try:
                # don't require a test set at train time,
                # but if there is one, get a sense of whether unks will be required
                self.test = self.tokenize_with_unks(os.path.join(path, testfname))
            except:
                pass
            #match any embeddings up with ids
            if embeddingfname:
                self.dictionary.match_embeddings()
            self.save_dict(vocab_file)
        else:
            # load pretrained model
            if vocab_file[-3:] == 'bin':
                self.load_dict(vocab_file)
            else:
                self.dictionary = Dictionary(embeddingfname, fasttext_loc, allowOOV)
                self.load_dict(vocab_file)
                if not os.path.exists('embeddings_'+vocab_file) and not interact_flag:
                    self.dictionary.match_embeddings()
                    self.save_dict(vocab_file)
            if test_flag:
                # test mode
                if multisentence_test_flag:
                    self.test = self.tokenize_with_unks(os.path.join(path, testfname))
                else:
                    self.test = self.sent_tokenize_with_unks(os.path.join(path, testfname))
            elif checkpoint_flag or predefined_vocab_flag:
                # load from a checkpoint
                self.train = self.tokenize_with_unks(os.path.join(path, trainfname))
                self.valid = self.tokenize_with_unks(os.path.join(path, validfname))


    def save_dict(self, path):
        """ Saves dictionary to disk """
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'wb') as file_handle:
                torch.save(self.dictionary, file_handle, pickle_module=dill)
        else:
            # Assume dict is plaintext
            with open(path, 'w') as file_handle:
                for word in self.dictionary.idx2word:
                    file_handle.write(word+'\n')
            #if embeddings exists let's save those too
            if len(self.dictionary.embeddings) != 0:
                embedding_path = 'embeddings_'+path
                with open(embedding_path, 'w') as f:
                    for embed in self.dictionary.embeddings:
                        f.write(' '.join(list(map(lambda x: str(x), embed)))+'\n')

    def load_dict(self, path, loadEmbedding=False):
        """ Loads dictionary from disk """
        assert os.path.exists(path), "Bad path: %s" % path
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'rb') as file_handle:
                fdata = torch.load(file_handle, pickle_module=dill)
                if isinstance(fdata, tuple):
                    # Compatibility with old pytorch LM saving
                    self.dictionary = fdata[3]
                self.dictionary = fdata
        else:
            # Assume dict is plaintext
            with open(path, 'r') as file_handle:
                for line in file_handle:
                    self.dictionary.add_word(line.strip())
            #if embeddings exists let's load those too
<<<<<<< HEAD
            if self.dictionary.w2vec is not None:
=======
            if self.dictionary.w2vec is not None and os.path.exists('embeddings_'+path) and loadEmbedding:
>>>>>>> sim
                with open('embeddings_'+path, 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        self.dictionary.embeddings.append(list(map(lambda x: float(x), line)))
                self.dictionary.embeddings = np.stack(self.dictionary.embeddings)

    def tokenize(self, path):
        """ Tokenizes a text file. """
        assert os.path.exists(path), "Bad path: %s" % path
        # Add words to the dictionary
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        if self.lower:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word.lower())
                        else:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word)

            # Tokenize file content
            with gzip.open(path, 'rb') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word.lower())
                                token += 1
                        else:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word)
                                token += 1
        else:
            with open(path, 'r') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        if self.lower:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word.lower())
                        else:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word.lower())
                                token += 1
                        else:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word)
                                token += 1
        return ids

    def tokenize_with_unks(self, path):
        """ Tokenizes a text file, adding unks if needed. """
        assert os.path.exists(path), "Bad path: %s" % path
        if path[-2:] == 'gz':
            # Determine the length of the corpus
            with gzip.open(path, 'rb') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with gzip.open(path, 'rb') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                # Convert OOV to <unk>
                                if word.lower() not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word.lower()]
                                token += 1
                        else:
                            for word in words:
                                # Convert OOV to <unk>
                                if word not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word]
                                token += 1
        else:
            # Determine the length of the corpus
            with open(path, 'r') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with open(path, 'r') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                # Convert OOV to <unk>
                                if word.lower() not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word.lower()]
                                token += 1
                        else:
                            for word in words:
                                # Convert OOV to <unk>
                                if word not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word]
                                token += 1
        return ids

    def sent_tokenize_with_unks(self, path):
        """ Tokenizes a text file into sentences, adding unks if needed. """
        assert os.path.exists(path), "Bad path: %s" % path
        all_ids = []
        sents = []
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb') as file_handle:
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        ids = self.convert_to_ids(words)
                        all_ids.append(ids)
        else:
            with open(path, 'r') as file_handle:
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        ids = self.convert_to_ids(words)
                        all_ids.append(ids)
        return (sents, all_ids)

    def encode(self, line, add_space_before_punct_symbol=False, lower=True):

        if lower:
            line = line.lower()

        if add_space_before_punct_symbol:
            punct = "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"
            #add space before punct
            line = line.translate(str.maketrans({key: " {0}".format(key) for key in punct}))
            #remove double spaces
            line = re.sub('\s{2,}', ' ', line)

        sentences = sent_tokenize(line)
        output = []
        for x, sent in enumerate(sentences):
            sent = sent.split(' ')
            if x == 0:
                sent = ['<eos>'] + sent

            #imagine we add a word is this really a sentence
            #If it's a sentence then sent_tokenize will 
            #generate two sentences
            #A bit hacky but it helps in parity with huggingface
            test_sent = ' '.join(sent + ['the'])
            if len(sent_tokenize(test_sent)) != 1:
                sent = sent + ['<eos>']

            output += list(self.convert_to_ids(sent).data.numpy())
        return output

    def decode(self, ids):
        words = list(map(lambda x: self.dictionary.idx2word[x], ids))
        return words

    def online_tokenize_with_unks(self, line):
        """ Tokenizes an input sentence, adding unks if needed. """
        all_ids = []
        sents = [line.strip()]

        words = ['<eos>'] + line.strip().split() + ['<eos>']

        ids = self.convert_to_ids(words)
        all_ids.append(ids)
        return (sents, all_ids)

    def convert_to_ids(self, words, tokens=None):
        if tokens is None:
            tokens = len(words)

        # Tokenize file content
        ids = torch.IntTensor(tokens)
        token = 0
        if self.lower:
            for word in words:
                # Convert OOV to <unk>
                if word.lower() not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word.lower()]
                token += 1
        else:
            for word in words:
                # Convert OOV to <unk>
                if word not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word]
                token += 1
        return(ids)
