import torch
from config import get_config
from model import CNNBiLSTM
from dataset.dataset import get_ds
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# def get_dl(data_path, tokenizer, seq_len, batch_size):
#     ds = get_ds(data_path, tokenizer, seq_len)
#     dl = DataLoader(ds, batch_size, shuffle=True)
#     return dl

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'Using device: "{device}"')
    model = CNNBiLSTM(config)
    # build datasets
    train_ds = get_ds(data_path=config.get('train_ds'), max_length=config.get('max_length'))
    val_ds = get_ds(data_path=config.get('val_ds'), max_length=config.get('max_length'))
    # build dataloaders
    train_dl = DataLoader(train_ds, batch_size=config.get('batch_size'), shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.get('batch_size'), shuffle=True)
    # build optimizer
    optimizer = Adam(model.parameters(), config.get('learning_rate'))
    # build loss function
    criterion = CrossEntropyLoss()

    for epoch in range(config.get('num_epochs')):
        for module in model.modules():
            module.to(device)
        model.train()
        batch_iterator = tqdm(train_dl, desc=f'Processing epoch {epoch+1:02d}')

        for batch in batch_iterator:
            tokens = batch['tokens'].to(device)
            # category_label.shape = (batch_size)
            # polarity_label.shape = (batch_size)
            # category_label = batch['category'].type(torch.LongTensor).to(device)
            polarity_label = batch['polarity'].type(torch.LongTensor).to(device)
            out = model(tokens)
            # category_pred.shape = (batch_size, 8)
            # polarity_pred.shape = (batch_size, 3)
            # category_pred = out.get('category')
            polarity_pred = out.get('polarity')

            # print(f'category_label.shape: {category_label.size()}')
            # print(f'polarity_label.shape: {polarity_label.size()}')
            # print()
            # print(f'category_pred.shape: {category_pred.size()}')
            # print(f'polarity_pred.shape: {polarity_pred.size()}')
            
            # loss_c = criterion(category_pred, category_label)
            loss_p = criterion(polarity_pred, polarity_label)
            # loss = loss_c + loss_p

            batch_iterator.set_postfix({'loss': f'{loss_p.item():6.3f}'})

            loss_p.backward()
            optimizer.zero_grad()
            optimizer.step()

        # with torch.no_grad():
        #     batch_iterator = tqdm(val_dl, desc=f'Evaluating epoch {epoch+1:02d}')
        #     for batch in batch_iterator:
        #         tokens = batch['tokens'].to(device)
        #         # category_label.shape = (batch_size)
        #         # polarity_label.shape = (batch_size)
        #         category_label = batch['category'].type(torch.LongTensor).to(device)
        #         polarity_label = batch['polarity'].type(torch.LongTensor).to(device)

        #         out = model(tokens)

        #         category_pred = out.get('category')
        #         polarity_pred = out.get('polarity')
                
        #         loss_c = criterion(category_pred, category_label)
        #         loss_p = criterion(polarity_pred, polarity_label)
        #         loss = loss_c + loss_p
        #         batch_iterator.set_postfix({'\033[32mval_loss\033[0m': f'\033[32m{loss.item():6.3f}\033[0m'})
            
        

if __name__ == '__main__':

    train(get_config())