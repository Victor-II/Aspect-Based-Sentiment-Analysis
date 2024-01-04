import spacy
import torch
from transformers import BartTokenizer, BartModel



class SpacyTokenizer():
    def __init__(self, load:str='en_core_web_sm'):
        self.tokenizer = spacy.load(load).tokenizer

    def __call__(self, sentence:str, text_only:bool=True):
        if text_only:
            return [token.text for token in self.tokenizer(sentence)]
        else:
            return [token for token in self.tokenizer(sentence)]



register = {
    'spacy_tokenizer': SpacyTokenizer()
}

def get_tokenizer(name:str='spacy_tokenizer'):
    return register.get(name)

if __name__ == '__main__':
    sentence = "The pot pie, pork chop and chicken were cleaned off the plates so well they didn't need washing."
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base')
    tokens = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True, padding='max_length', max_length=70)])
    embeddings = model(tokens)[0]
    print(embeddings.size())
   