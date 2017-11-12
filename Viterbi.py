"""
输入的文字序列即为观测序列，观测序列的长度即为t,
初始状态向量(1*n):词性的概率向量，{vp:, vn:, vb:}。
状态转移矩阵(n*n):从该位置的词性到下一位置另一词性转移的概率。
发射概率矩阵(n*v):已知为该词性的条件下，是某个词的概率。
"""
from collections import defaultdict
import numpy as np
from nltk import word_tokenize

vocabulary = dict()
the_pos = dict()  # 词性在状态转移矩阵的位置
all_pairs = dict()  # 所有的词性标注对
start_probability = defaultdict()
state_transition = defaultdict()
emission_probability = defaultdict(dict)
last_token = ''
for token, word in all_pairs.items():
    if token not in the_pos:
        the_pos[token] = len(the_pos)
    if word not in vocabulary:
        vocabulary[word] = len(vocabulary)
    start_probability[token] += 1  # 某词性下词数目加1
    emission_probability[token][word] += 1  # 在词性确定的条件下某个词的概率(数目)
    state_transition[(last_token, token)] += 1  # 从t时刻的某一词性转移到t+1另一词性的概率
    last_token = token
words_num = sum(start_probability.values())
start_probability = {k: (v / words_num) for k, v in start_probability.items()}
the_pos_num = len(the_pos)
state_probability_matrix = np.zeros((the_pos_num, the_pos_num))
for every_transition in state_transition:
    row = the_pos[every_transition[0]]
    column = the_pos[every_transition[1]]
    state_probability_matrix[row][column] = state_transition[every_transition]
for every_row in state_probability_matrix:
    every_row /= sum(every_row)
for every_pos in emission_probability:
    sss = sum(every_pos.values())
    for every_value in emission_probability[every_pos]:
        every_value /= sss

string_cluster = ''
word_list = word_tokenize(string_cluster)
words_num = len(word_list)
state_num = len(start_probability)
time_series_list = []
first_layer = [start_probability[every_pos] for every_pos in the_pos]
time_series_list.append(first_layer)
for n in range(1, words_num + 1):
    for current_state in the_pos:
        for last_state in the_pos:
            last_state_prob = time_series_list[n - 1][the_pos[last_state]]
            prob_from_last_to_now = state_probability_matrix[the_pos[last_state]][the_pos[current_state]]
            current_emission = emission_probability[current_state][word_list[n]]
            temp = last_state_prob*prob_from_last_to_now*current_emission
