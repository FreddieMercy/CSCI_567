import numpy as np
from hmm import HMM


def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################

    for k in unique_words.keys():
        word2idx[k] = len(word2idx)

    for i in range(S):
        tag2idx[tags[i]] = i

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################

    for i in train_data:
        pi[tag2idx[i.tags[0]]] += 1
        B[tag2idx[i.tags[0]]][word2idx[i.words[0]]] += 1
        for j in range(1, len(i.tags)):
            A[tag2idx[i.tags[j - 1]]][tag2idx[i.tags[j]]] += 1
            B[tag2idx[i.tags[j]]][word2idx[i.words[j]]] += 1

    total_pi = np.sum(pi)
    if total_pi != 0:
        pi /= total_pi

    total_A = np.sum(A, axis=1)
    total_B = np.sum(B, axis=1)

    for i in range(S):
        if total_A[i] != 0:
            A[i] /= total_A[i]
        if total_B[i] != 0:
            B[i] /= total_B[i]

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    S = len(tags)
    E6 = 1e-6
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################

    #model = model_training(test_data, tags)

    for i in test_data:
        for item in i.words - model.obs_dict.keys():
            model.obs_dict[item] = len(model.obs_dict)
            newValue = np.zeros(S)
            newValue[:] = E6
            model.B = np.append(model.B, newValue.reshape(-1, 1), axis=1)
        tagging.append(model.viterbi(i.words))

    return tagging


# DO NOT MODIFY BELOW
def get_unique_words(data):
    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
