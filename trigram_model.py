import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    sequence.insert(0, "START")
    sequence.append("STOP")

    if n >= 3:
        for i in range(n - 2):  # 2 starts for n=3
            sequence.insert(0, "START")

    ngrams_list = []
    ngrams = []

    for i in range(n):
        ngrams.append(sequence[i:])

    ngrams_zip = zip(*[ngrams[i] for i in range(len(ngrams))])

    for w1 in ngrams_zip:
        ngrams_list.append(w1, )

    return ngrams_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)




    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.total_no_words = 0
        self.total_no_sentences = 0
        self.total_no_vocab = 0

        self.unigramcounts = {}
        self.bigramcounts = {} 
        self.trigramcounts = {}

        uni = defaultdict(int)
        big = defaultdict(int)
        trig = defaultdict(int)
        for sentence in corpus:
            self.total_no_sentences += 1
            for word in get_ngrams(sentence, 1):
                uni[word] += 1
                self.total_no_words += 1 #in unigram model
            for word in get_ngrams(sentence, 2):
                big[word] += 1
            for word in get_ngrams(sentence, 3):
                trig[word] += 1

        self.total_no_vocab = len(uni.keys()) #total number of unique words
        self.unigramcounts = uni
        self.bigramcounts = big
        self.trigramcounts = trig

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[(trigram[0], trigram[1],)]:
            return self.trigramcounts[trigram] / self.bigramcounts[(trigram[0], trigram[1],)]
        else:
            return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[(bigram[0],)] == 0:
            return 0.0

        return self.bigramcounts[bigram] / self.unigramcounts[(bigram[0],)]


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        return self.unigramcounts[unigram] / self.total_no_words


    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return #result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0


        bigram = (trigram[0], trigram[1],)
        unigram = (trigram[0],)

        STP = (lambda1 * self.raw_trigram_probability(trigram)) +(lambda2 * self.raw_bigram_probability(bigram)) + (lambda3 * self.raw_unigram_probability(unigram))
        return STP

        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        log_probabilities = 0

        for tuple in trigrams:
            smoothed_probability = self.smoothed_trigram_probability(tuple)
            log_format = math.log2(smoothed_probability)
            log_probabilities += log_format

        return log_probabilities

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M = 0
        sum = 0

        for sentence in corpus:
            M += len(sentence)
            SLP = self.sentence_logprob(sentence) #returns log probability of the sentence
            sum += SLP
        perplexity = 2**(-sum/M)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1) # 'train_high.txt'
        model2 = TrigramModel(training_file2) # 'train_low.txt'

        total = 0
        correct = 0  #correct predictions = true positive + true negative

        for f in os.listdir(testdir1): # loop over every text file in the directory
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            # actual score == high
            if pp1 < pp2:
                #model 1 made correct prediction
                #true positive case
                correct += 1

            total += 1

        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))

            #actual score == low
            if pp2 < pp1:
                #model 2 made correct prediction
                # true negative case
                correct += 1

            total += 1

        accuracy = (correct/total) * 100 #in percentage
        return accuracy

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt.

    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', 'hw1_data/ets_toefl_data/test_high', 'hw1_data/ets_toefl_data/test_low')
    print(acc)


