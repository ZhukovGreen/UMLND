import string

sample_memo = """
Milt, we're gonna need to go ahead and move you downstairs into storage B. We 
have some new people coming in, and we 
need all the space we can get. So if you could just go ahead and pack up your 
stuff and move it down there, that would 
be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you 
want to, go ahead and wear a Hawaiian shir
t and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come
 in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in 
tomorrow. So if you could be here around 
9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna
 need you to go ahead and come in on Sunday
too, kay. We ahh lost some people this week and ah, we sorta need to play catch
 up.
"""


#   Maximum Likelihood Hypothesis
#
#
#   In this quiz we will find the maximum likelihood word based on the precedi
# ng word
#
#   Fill in the NextWordProbability procedure so that it takes in sample text
# and a word,
#   and returns a dictionary with keys the set of words that come after, whose
#  values are
#   the number of times the key comes after that word.
#
#   Just use .split() to split the sample_memo text into words separated by
# spaces.

def NextWordProbability(sampletext, word):
    """

    Parameters
    ----------
    sampletext : str
    word : str

    Returns
    -------
    x : dict
    """
    sampletext = sampletext.translate(
        str.maketrans({k: None for k in string.punctuation}))
    words = [w.lower().replace('\n', '') for w in sampletext.split(' ')]
    words = [words[idx + 1] for idx, word_ in enumerate(words) if
             word_ == word and idx < len(words) - 1]
    word_freq_dict = dict(
        zip(words, [words.count(word) for word in words]))
    return word_freq_dict


print(NextWordProbability(sample_memo, 'and'))
