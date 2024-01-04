import torch
import torch.nn as nn
from transformers import BartModel, BartConfig
from torch.autograd import Variable
from config import get_config


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    
class BartEmbedding(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.model = BartModel.from_pretrained(pretrained_path)
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.model(x)[0]

class CNNModel(nn.Module):
    def __init__(self, max_length, kernel_size, padding):
        super().__init__()
        # self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv1d = nn.Conv1d(in_channels=max_length, out_channels=max_length, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(num_features=max_length)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        return x.squeeze(dim=1) # x = (batch_size, seq_len, embedding_dim)

class BiLSTMModel(nn.Module):
    def __init__(self, embedding_dim, lstm_units, lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=lstm_units, 
                            num_layers=lstm_layers, 
                            bidirectional=True, 
                            batch_first=True)
        self.relu = nn.ReLU()
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units

    def init_hidden(self, batch_size):
        h, c = (
                Variable(torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_units)),
                Variable(torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_units))
                )
        return h.to('cuda'), c.to('cuda')
    
    def forward(self, x):
        batch_size = x.size()[0]
        out, (h, c) = self.lstm(x, self.init_hidden(batch_size))
        out = self.relu(out[:, -1, :]) # (batch_size, lstm_units * 2)
        return out
        

class LinearModel(nn.Module):
    def __init__(self, lstm_units, hidden_dim, num_category, num_polarity, dropout):
        super().__init__()
        self.linear = nn.Linear(lstm_units*2, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_c = nn.Linear(hidden_dim, num_category)
        self.linear_p = nn.Linear(hidden_dim, num_polarity)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = (batch_size, lstm_units * 2)
        x = self.linear(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # c = self.linear_c(x)
        p = self.linear_p(x)
        # c = (batch_size, num_category)
        # p = (batch_size, num_polarity)
        return p

class CNNBiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = BartEmbedding(config.get('embedding_pretrained_path'))
        # self.embedding = InputEmbedding(vocab_size=config.get('vocab_size'),
        #                                 embedding_dim=config.get('embedding_dim'))
        
        # self.dropout = nn.Dropout2d(0.3)

        self.cnn = CNNModel(max_length=config.get('max_length'),
                            kernel_size=config.get('kernel_size'),
                            padding=config.get('padding'))
        
        self.bilstm = BiLSTMModel(embedding_dim=config.get('embedding_dim'),
                                  lstm_units=config.get('lstm_units'),
                                  lstm_layers=config.get('lstm_layers'))
        
        self.linear = LinearModel(lstm_units=config.get('lstm_units'),
                                  hidden_dim=config.get('linear_hidden_dim'),
                                  num_category=config.get('num_category'),
                                  num_polarity=config.get('num_polarity'),
                                  dropout=config.get('dropout'))

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # x = self.dropout(x)
        x = self.cnn(x)        # (batch_size, seq_len, embedding_dim)
        x = self.bilstm(x)     # (batch_size, lstm_units)
        p = self.linear(x)  # (batch_size, num_category), (batch_size, num_polarity)
        return {'polarity': p}
        # return {'category': c, 'polarity': p}
    
if __name__ == '__main__':

    config = get_config()
    # input = (batch_size=8, seq_len=16, embedding_dim=64)
    input = torch.zeros(size=[config.get('batch_size'), config.get('max_length')], dtype=torch.int)
    model = CNNBiLSTM(config=config)
    
    out = model(input)
    print(out.get('category').size())
    print(out.get('polarity').size())
 