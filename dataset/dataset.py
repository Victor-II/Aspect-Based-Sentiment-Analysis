import torch
import spacy
from torch.utils.data import Dataset
# from tokenizer import get_tokenizer
from dataset.utils import build_mapping, get_preprocessed_data, build_category_polarity_mapping
from dataset.constants import UNK, PAD, CLS, SEP, SEQ_LEN
from transformers import BartTokenizer


class ACSA_DS(Dataset):
    def __init__(self, data_path:list, max_length:int, tokenizer:str, pretrained_path:str) -> None:
        super().__init__()

        self.data = get_preprocessed_data(data_path)
        self.tokenizer = tokenizer
        self.pretrained_path = pretrained_path
        self.max_length = max_length
        self.category_polarity_mapping = build_category_polarity_mapping(self.data)

    def __len__(self):
        return len(self.data)
    
    def _bart_tokenizer(self, sentence):
        tokenizer = BartTokenizer.from_pretrained(self.pretrained_path)
        tokens = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True, padding='max_length', max_length=self.max_length)])
        return tokens.squeeze(dim=0)
    
    # def _spacy_tokenizer(self, sentence):
    #     tokenizer = spacy.load(self.pretrained_path).tokenizer
    #     mapping = build_mapping(self.data, tokenizer, special_tokens=[PAD, UNK, SEP, CLS])
    #     print(mapping['word2index'])
    #     tokens = [index for index in [mapping['word2index'][token] for token in tokenizer(sentence)]]
    #     padding = self.max_length - len(tokens) -2

    #     tokens = torch.cat(
    #         [
    #             torch.tensor([self.mapping['word2index'][CLS]], dtype=torch.int),
    #             torch.tensor(tokens, dtype=torch.int),
    #             torch.tensor([self.mapping['word2index'][SEP]], dtype=torch.int),
    #             torch.tensor([self.mapping['word2index'][PAD]] * padding, dtype=torch.int)
    #         ]
    #     )
    #     return tokens
    
    def _tokenize(self, sentence, tokenizer):
        # if tokenizer == 'spacy':
        #     return self._spacy_tokenizer(sentence)
        if tokenizer == 'bart':
            return self._bart_tokenizer(sentence)
            
    def __getitem__(self, index):
        sentence, category, polarity = self.data[index]
        # Map text
        tokens = self._tokenize(sentence, self.tokenizer)
        # Map categories
        category_tensor = torch.tensor(self.category_polarity_mapping.get('category2index').get(category), dtype=torch.int)
        # Map polarities
        polarity_tensor = torch.tensor(self.category_polarity_mapping.get('polarity2index').get(polarity), dtype = torch.int)

        assert tokens.size(0) == self.max_length, f'tokens.size(0) = {tokens.size(0)}'
        return {'tokens': tokens, 'category': category_tensor, 'polarity': polarity_tensor}
    
def get_ds(data_path:str, max_length:int, tokenizer:str='bart', pretrained_path:str='facebook/bart-base'):
    return ACSA_DS(data_path=data_path,
                   max_length=max_length,
                   tokenizer=tokenizer,
                   pretrained_path=pretrained_path)  

if __name__ == '__main__':
    pass
    

   
    






    # print(ds[2])
    
    

    


    
