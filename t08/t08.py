from datasets_tut8 import datasets
import numpy as np

trec07 = datasets.load_trec07()

x = trec07.data
y = trec07.target


def count_spam():
    num_spam = 0

    for i in range(len(y)):
        if y[i] == 'spam':
            num_spam += 1

    return num_spam

p_spam = count_spam() / len(y)
p_ham = 1 - p_spam

input_word = 'replica'


def prob_word_given_label (wordStr, labelStr, subjects, labels):

    total_with_label = 0;
    total_with_word = 0;

    for i in range(len(subjects)):
        if labels[i] == labelStr:
            total_with_label += 1
            for j in range(len(subjects[i])):
                if subjects[i][j] == wordStr:
                    total_with_word += 1

    if total_with_word != 0:
        return total_with_label / total_with_word

    return 1e-8


def run (num_train, num_test):
    x_train = x[0:num_train - 1]
    y_train = y[0:num_train - 1]
    x_test = x[num_train:num_train+num_test]
    y_test = y[num_train:num_train+num_test]

    num_correct = 0;

    for i in range(len(x_test)):
        p_this_spam = p_spam
        p_this_ham = p_ham
        for j in range(len(x_test[i])):
            p_this_spam *= prob_word_given_label(x_test[i][j], 'spam', x_train, y_train)
            p_this_ham *= prob_word_given_label(x_test[i][j], 'ham', x_train, y_train)
        if ((p_this_spam > p_this_ham) and y_test[i] == 'spam') or ((p_this_ham > p_this_spam) and y_test[i] == 'ham'):
            num_correct += 1

    correct = num_correct / len(x_test)
    print("p_spam: " + str(p_spam))
    print("p_ham: " + str(p_ham))
    print ("correct: " + str(correct))




run(100, 100)