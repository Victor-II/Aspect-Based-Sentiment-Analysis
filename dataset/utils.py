import xml.etree.ElementTree as ET
import spacy
import json
import string
import torch
from dataset.constants import PAD, UNK, CLS, SEP
from dataset.split_sentence import split_sentence


def get_word_counts(data, tokenizer):
    '''
    Receives data in the form of a list of tuples (text, category, polarity) as returned by get_raw_data
    Returns a dict with key: token, value: token_frequency
    '''
    word_counts = dict()
    for element in data:
        tokens = tokenizer(element[0])
        for token in tokens:
            if not token in word_counts.keys(): word_counts[token] = 1
            else: word_counts[token] += 1
    return word_counts

def build_mapping(data, tokenizer, special_tokens:list=[PAD, UNK, SEP, CLS], min_frequency:int=1) -> None:
    # Build word2index
    word_counts = get_word_counts(data, tokenizer)
    words = sorted([word for word in word_counts.keys()], key=lambda word: word_counts[word])
    categories, polarities = get_categories_and_polarities(data)
    word2index = dict()
    for special_token in special_tokens:
        word2index[special_token] = len(word2index)
    for word in words:
        if word_counts[word] >= min_frequency: word2index[word] = len(word2index)
        else: word2index[word] = word2index[UNK]

    # Build category2index
    category2index = dict()
    for category in categories:
        category2index[category] = len(category2index)

    # Build polarity2index
    polarity2index = dict()
    for polarity in polarities:
        polarity2index[polarity] = len(polarity2index)

    return {'word2index': word2index, 'category2index': category2index, 'polarity2index': polarity2index}
    # with open(file_path+'word2index.json', "w") as f:
    #     json.dump(word2index, f)
    # with open(file_path+'category2index.json', "w") as f:
    #     json.dump(category2index, f)
    # with open(file_path+'polarity2index.json', "w") as f:
    #     json.dump(polarity2index, f)

def build_category_polarity_mapping(data):
    categories, polarities = get_categories_and_polarities(data)
    category2index = dict()
    for category in categories:
        category2index[category] = len(category2index)

    # Build polarity2index
    polarity2index = dict()
    for polarity in polarities:
        polarity2index[polarity] = len(polarity2index)
    
    return {'category2index': category2index, 'polarity2index': polarity2index}

def get_raw_data(path:str) -> list:
    '''
    Parses data from an XML file with structure <sentences>{<sentence>{<text>, <aspectCategories>{category, polarity}}}
    Returns a list of tuples (text, category, polarity)
    '''
    tree = ET.parse(path)
    sentences = tree.getroot()
    data = []
    for sentence in sentences:
        text = sentence.find('text').text
        if text is None:
            continue
        category_polarity_pairs = [[aspect_category.get('category'), aspect_category.get('polarity')] for aspect_category in sentence.find('aspectCategories')]
        data.append([text, category_polarity_pairs])

    return data

def split_sentences(data:list, policy:str='drop') -> list:
    '''
    TODO
    '''
    data_split = []
    if policy == 'drop':
        for entry in data:
            entry_split = [[clause, cp_pair[0], cp_pair[1]] for clause, cp_pair in zip(split_sentence(entry[0]), entry[1])]
            data_split.extend(entry_split)
    return data_split

def remove_punctuation(data):
    for elem in data:
        elem[0] = elem[0].translate(str.maketrans('', '', string.punctuation))
    return data

def get_categories_and_polarities(data):
    categories = set()
    polarities = set()
    for elem in data:
        categories.add(elem[1])
        polarities.add(elem[2])

    return list(categories), list(polarities)

def get_category_and_polarity_counts(data):
    category_count = dict()
    polarity_count = dict()
    for _, category, polarity in data:
        if category in category_count.keys(): category_count[category] += 1
        else: category_count[category] = 1

        if polarity in polarity_count.keys(): polarity_count[polarity] += 1
        else: polarity_count[polarity] = 1

    return category_count, polarity_count

def get_max_seq_len(data, tokenizer):
    max_seq_len = 0
    for elem in data:
        tokens = tokenizer(elem[0])
        max_seq_len = max(max_seq_len, len(tokens))

    return max_seq_len

def get_preprocessed_data(path):
    data = get_raw_data(path)
    data = remove_punctuation(data)
    data = split_sentences(data)
    return data

if __name__ == '__main__':

    data = get_preprocessed_data('/home/victor-ii/Desktop/Research_ABSA/Task_1/data/mams/mams_acsa/test.xml')
    
    print(data[0][0])