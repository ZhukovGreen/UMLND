# ------------------------------------------------------------------

import itertools
#
#   Bayes Optimal Classifier
#
#   In this quiz we will compute the optimal label for a second missing word
#   in a row
#   based on the possible words that could be in the first blank
#
#   Finish the procedure, LaterWords(), below
#
#   You may want to import your code from the previous programming exercise!
#
import string
from pprint import pprint

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We 
have some new people coming in, and we need all the space we can get. So if you
could just go ahead and pack up your stuff and move it down there, that would 
be terrific, OK? Oh, and remember: next Friday... is Hawaiian shirt day. So, 
you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come 
in on Sunday, too... Hello Peter, whats happening? Ummm, I'm gonna need you to 
go ahead and come in tomorrow. So if you could be here around 9 that would be 
great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go 
ahead and come in on Sunday too, kay. We ahh lost some people this week and ah,
we sorta need to play catch up. move you to move you to
'''

corrupted_memo = '''
Yeah, I'm gonna xxx you to go ahead xxx xxx complain about this. Oh, and if you
could xxx xxx and sit at the kids' table, that'd be xxx 
'''

data_list = [w.lower().replace('\n', '') for w in sample_memo.split(' ')]

words_to_guess = ["ahead", "could"]


def LaterWords(sample, word, distance):
    """
    @param sample: a sample of text to draw from
    @param word: a word occurring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next
        word, 2 for the word after that)
    @returns: a single word which is the most likely possibility
    """
    sample = sample.translate(
        str.maketrans({k: None for k in string.punctuation}))
    # sample = sample.translate(None, string.punctuation)
    words = [w.lower().replace('\n', '') for w in sample.split(' ')]
    # TODO:
    # Given a word, collect the relative probabilities of possible following
    # words from @sample. You may want to import your code from the maximum
    # likelihood exercise.

    # TODO:
    # Repeat the above process--for each distance beyond 1, evaluate
    # the words that might come after each word, and combine them weighting by
    # relative probability into an estimate of what might appear next.

    words_matrix = [tuple(y)[:distance] for x, y in
                    itertools.groupby(words, lambda z: z == word) if not x][1:]
    words_matrix = [len(words_matrix) * [word]] + [[w[i] for w in words_matrix]
                                                   for i in range(distance)]

    def prob_of_a_word(current_layer_num, search_word):
        if current_layer_num == 0:
            return 1
        current_layer = words_matrix[current_layer_num]
        previous_layer = words_matrix[current_layer_num - 1]
        prob_word_in_current = 0
        for unique_word_in_previous_layer in set(previous_layer):
            cond_prob_word_in_current = 0
            for word_idx, word_in_current_layer in enumerate(current_layer):
                if word_in_current_layer == search_word and previous_layer[
                    word_idx] == unique_word_in_previous_layer:
                    cond_prob_word_in_current += 1.0
            cond_prob_word_in_current = \
                cond_prob_word_in_current / previous_layer.count(
                    unique_word_in_previous_layer)
            if cond_prob_word_in_current:
                prob_word_in_current += cond_prob_word_in_current * prob_of_a_word(
                    current_layer_num - 1, unique_word_in_previous_layer)
        return prob_word_in_current

    return max([(word_, prob_of_a_word(distance, word_)) for word_ in
                words_matrix[distance]], key=lambda x: x[1])[0]


words = corrupted_memo.split()
res = []
for idx, word in enumerate(words):
    if word == 'xxx':
        word = LaterWords(sample_memo, words[idx - 1], 1)
        words[idx] = word
    res.append(word)
pprint(' '.join(res))
