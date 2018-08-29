import numpy as np
import os
import re
import pickle
import gzip
import tarfile
import sys


__author__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"

class datasets:

    def __init__(self, N=None):

        picklefile = 'trec07p.pickle'
        if not os.path.exists(picklefile):
            sys.stdout.write('Reading data for the first time...')
            sys.stdout.flush()
            datadir = 'trec07p'
            if not os.path.exists(datadir):
                filepath = 'trec07p.tgz'
                if not os.path.exists(filepath):
                    raise FileNotFoundError('trec07p.tgz not found in the project directory.')

                tarfile.open(filepath, 'r:gz').extractall('./')

            labels, subjects = self.read_from_dir(dir_name=datadir)

            # Save and archive the model
            with gzip.open(picklefile, 'w') as f:
                pickle.dump((labels, subjects), f)
            sys.stdout.write('done\n')
            sys.stdout.flush()

        else:
            with gzip.open(picklefile) as f:
                (labels,subjects) = pickle.load(f)

        subjects = np.array(subjects)
        labels = np.array(labels)

        if N is None or N>len(subjects):
            N = len(subjects)

        Is = np.argwhere(labels=='spam')[:,0]
        Ih = np.argwhere(labels=='ham')[:,0]

        p = 0.35
        Nspam = int(np.round(N*p))

        if Nspam > len(Is):
            Nspam = len(Is)


        Nham = N-Nspam
        if Nham > len(Ih):
            Nham = len(Ih)

        Nnew = int(np.round(float(Nham)/(1-p)))
        if Nnew < N:
            Nspam = int(np.round(p*Nnew))


        Is = Is[:Nspam]
        Ih = Ih[:Nham]
        I = np.sort(np.concatenate([Is, Ih]))

        subjects = subjects[I]
        labels = labels[I]

        np.random.seed(0)
        I = np.random.permutation(len(labels))

        self.data = subjects[I]
        self.target = labels[I]



    def read_from_dir(self,dir_name):

        label_file = os.path.join(dir_name, 'full','index')

        with open(label_file) as f:
           label_strings = f.readlines()

        labels = []
        subjects = []

        for label_s in label_strings:
            match = re.search('inmail.(\d+)', label_s)
            if match:
                l = label_s.split()[0]
                data_file = os.path.join(dir_name,'data',match.group(0))

                wordlist = []
                with open(data_file,encoding="utf8", errors='ignore') as f:
                    firstLine = False

                    for line in f:
                        if firstLine:
                            line = line.split()
                            if line[0] == 'From':
                                wordlist.append(line[1])
                            else:
                                print("What gives?")
                            firstLine = False
                            continue

                        match = re.search('(?:Subject).*', line)
                        if match:
                            subject_text = match.group(0).partition('Subject: ')[-1]
                            wordstr = subject_text.rstrip()
                            for words in wordstr.split():
                                skipWord = False
                                while len(words)>0:
                                    if words[-1] == ',':
                                        words = words[:-1]
                                    elif words[-1] == '@':
                                        words = words[:-1]
                                    elif words[-1] == ';':
                                        words = words[:-1]
                                    elif words == '[R]' or words == 'Re:' or words=='Fwd:' or words=='&' or words=='/' or words == 'RE':
                                        skipWord = True
                                        break
                                    elif words[-1] == '"':
                                        words = words[:-1]
                                    elif words[-1] == ')':
                                        words = words[:-1]
                                    elif words[0] == '"':
                                        words = words[1:]
                                    elif words[0] == '(':
                                        words = words[1:]
                                    elif words[0] == '#':
                                        words = words[1:]
                                    elif words[-1] == '.' or words[-1] == '?' or words[-1] == ':' or words[-1] == '!':
                                        words = words[:-1]
                                    else:
                                        break

                                if skipWord or len(words) == 0:
                                    continue

                                wordlist.append(words.lower())

                            if len(wordlist) > 0:
                                subjects.append(np.array(wordlist))
                                labels.append(l)

                            break


        return labels, subjects


    @staticmethod
    def load_trec07(N=None):
        # Read the data
        return datasets(N=N)





