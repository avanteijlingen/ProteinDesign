# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:09:53 2023

@author: Alex
"""
import numpy as np
import json, os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import euclidean_distances, mean_squared_error, r2_score

import torch
import torch.nn as nn
import tqdm, math
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1], pos
        # div_term [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^{2i/d_model}
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len,1,d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: [seq_len, batch_size, d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    :return:
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self,input_Q, input_K, input_V, attn_mask):
        '''
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B,S,D) - proj -> (B,S,D_new) -split -> (B, S, H, W) -> trans -> (B,H,S,W)

        # 分解为MultiHead Attention
        Q = self.W_Q(input_Q).view(batch_size,-1, n_heads, d_k).transpose(1,2) # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size,-1, n_heads, d_k).transpose(1,2) # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size,-1, n_heads, d_v).transpose(1,2) # V:[batch_size, n_heads, len_v(=len_k, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask: [batch_size,n_heads, seq_len, seq_len]

        # [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q,K,V, attn_mask)
        context = context.transpose(1,2).reshape(batch_size, -1, n_heads * d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)

        return nn.LayerNorm(d_model).to(device)(output+residual),attn # Layer Normalization

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output+residual) #[batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,enc_inputs, enc_self_attn_mask):
        '''
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return:
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :return:
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) # [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
    
class Classifier(nn.Module):
    def __init__(self, input_dim, Min, Max):
        super().__init__()
        last_layer = 1
        hidden_layer = [512,256,64,32,last_layer]
        #hidden_layer = [batch_size,256,64,32,last_layer]
        self.sigmoid = nn.Sigmoid()
        self.e1 = nn.Linear(input_dim, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.e5 = nn.Linear(hidden_layer[3], hidden_layer[4])
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm0 = nn.BatchNorm1d(hidden_layer[0])
        self.batchnorm1 = nn.BatchNorm1d(hidden_layer[1])
        self.batchnorm2 = nn.BatchNorm1d(hidden_layer[2])
        self.batchnorm3 = nn.BatchNorm1d(hidden_layer[3])
        self.sigmoid = nn.Sigmoid()
        self.max = Max
        self.min = Min
    
        
    def forward(self,dec_input):
        h_1 = F.leaky_relu(self.batchnorm0(self.e1(dec_input)), negative_slope=0.05, inplace=True)
        h_1 = self.dropout(h_1)
        h_2 = F.leaky_relu(self.batchnorm1(self.e2(h_1)), negative_slope=0.05, inplace=True)
        h_2 = self.dropout(h_2)
        h_3 = F.leaky_relu(self.e3(h_2), negative_slope=0.1, inplace=True)
        h_3 = self.dropout(h_3)
        h_4 = F.leaky_relu(self.e4(h_3), negative_slope=0.1, inplace=True)
        y = self.e5(h_4)

        return self.sigmoid(y)*(self.max-self.min)+self.min
        
class Transformer(nn.Module):
    def __init__(self, Min, Max):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Classifier(src_len*d_model, Min, Max).to(device)
        #self.max = 2.89703
        #self.min = 0.959986
        self.max = Max
        self.min = Min

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''
        enc_outputs,_ = self.encoder(enc_inputs)
        # dec_inputs = enc_outputs[:,0,:].squeeze(1)
        dec_inputs = torch.reshape(enc_outputs,(enc_outputs.shape[0],-1))
        pred = self.decoder(dec_inputs)

        return pred.float()

# =============================================================================
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# #device = "cpu"
# print("Device:", device)
# =============================================================================
device = torch.device('cpu')
d_model = 512 # Embedding size
src_vocab_size = 21
lr = 0.002
n_layers = 6 # number of Encoder and Decoder Layer
d_k = 64 # dimension of K(=Q), V
d_v = 64 # dimension of K(=Q), V
n_heads = 8 # number of heads in Multi-Head Attention
d_ff = 2048 # FeedForward dimension
src_len = 70
batch_size = 100 #1024
src_vocab = {'Empty':0, 'A': 1, 'C': 2,'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 
                'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 
                'T': 17, 'V': 18, 'W': 19, 'Y': 20}

def make_data(features: list, src_len: int) -> torch.LongTensor:
    """
    Parameters
    ----------
    features : list
        The spike interface sequence as a list of characters.
    src_len : int
        The max length we expect the interface sequence to be (70).

    Returns
    -------
    torch.LongTensor
        Position encoded sequence.

    """
    enc_inputs = []
    for i in range(len(features)):
        enc_input = [[src_vocab[n] for n in list(features[i])]]
        while len(enc_input[0])<src_len:
            enc_input[0].append(0)
        enc_inputs.append(enc_input)
    return torch.LongTensor(enc_inputs)

def encode_data(Data: dict, target: str, device):
    X = np.ndarray((0, src_len), dtype=np.int64)
    Y = np.ndarray((0, ))
    for key in Data[target]:
        if len(key) != src_len:
            continue
        x = make_data([list(key)], len(key)).numpy()
        X = np.vstack((X, x.reshape(1, src_len)))
        Y = np.hstack((Y, Data[target][key]["BindingEnergy"]))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=95)
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, Y.min(), Y.max()

def train_transformer_model(device, train_dataloader, test_dataloader, Min_val, Max_val, checkpoint_fname = "best.pt"):
    mse_sum = torch.nn.MSELoss(reduction='sum')
    model = Transformer(Min=Min_val, Max=Max_val)
    model.to(device)
    loss_function = nn.MSELoss()
    #loss_function = r2_score
    SGD = torch.optim.SGD(model.parameters(), lr=lr)
    SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.2, patience=1000, threshold=0, 
                                                               min_lr = 0.000001, verbose=True)
    # Load model if exists
    if os.path.exists(checkpoint_fname):
        print("Loading from:", checkpoint_fname)
        if device.type == "cpu":
            checkpoint = torch.load(checkpoint_fname, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_fname)
        model.load_state_dict(checkpoint)
        #SGD.load_state_dict(checkpoint['SGD'])
        #SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])
    history = {"Test loss": [], "Train loss": []}
    for epoch in tqdm.tqdm(range(SGD_scheduler.last_epoch, SGD_scheduler.last_epoch+210)):
        LossSum = None
        model.train()
        total_mse = 0
        count = 0
        for X_train, y_train in train_dataloader:
            SGD.zero_grad()
            pred = model(X_train)
            pred = pred.flatten()
            loss = torch.sqrt(loss_function(pred, y_train))
            total_mse += mse_sum(pred, y_train).item()
            count += pred.size(0)
            loss.backward()
            SGD.step()
        loss = math.sqrt(total_mse / count)
        history["Train loss"].append(loss)
        SGD_scheduler.step(loss)
        
        model.eval()
        train_pred = np.ndarray((0,))
        train_measured = np.ndarray((0,))
        for X_train, y_train in train_dataloader:
            predict = model(X_train).reshape(-1)
            pred_y = predict.cpu().detach().numpy().flatten()
            train_measured = np.hstack((train_measured, y_train.cpu().detach().numpy().flatten()))
            train_pred = np.hstack((train_pred, pred_y))
            
        total_mse = 0
        count = 0
        true_all = np.ndarray((0,))
        pred_all = np.ndarray((0,))
        for X_test, y_test in test_dataloader:
            predict = model(X_test).reshape(-1)
            pred_y = predict.cpu().detach().numpy().flatten()
            pred_all = np.hstack((pred_all, pred_y))
            true_all = np.hstack((true_all, y_test.cpu().detach().numpy().flatten()))
            total_mse += mse_sum(predict, y_test).item()
            count += predict.size(0)
        test_loss = math.sqrt(total_mse / count)
        r2 = r2_score(pred_all, true_all)
        
        if epoch > 0:
            if test_loss < min(history["Test loss"]):
                torch.save(model.state_dict(), checkpoint_fname)
                plt.plot([Min_val, Max_val], [Min_val, Max_val], lw=1, color="black")
                plt.scatter(train_measured, train_pred, s=11, color="blue", alpha=0.8, label="Training")
                plt.scatter(true_all, pred_all, s=11, color="orange", alpha=0.8, label="Testing")
                plt.xlabel("Measured AIMNet2 Binding Energy (kcal/mol)")
                plt.ylabel("Predicted AIMNet2 Binding Energy (kcal/mol)")
                #plt.title(f"Best Test RMSE: {round(test_loss, 2)}, r2: {round(r2, 1)}, EPOCH: {epoch}")
                plt.title(f"Best Test RMSE: {round(test_loss, 2)}, r2: {round(r2, 1)}, {target}")
                plt.show()
        history["Test loss"].append(test_loss)
    return model, history

def transformer_plot(device, train_dataloader, test_dataloader, Min_val, Max_val, checkpoint_fname = "best.pt"):
    mse_sum = torch.nn.MSELoss(reduction='sum')
    model = Transformer(Min=Min_val, Max=Max_val)
    model.to(device)

    print("Loading from:", checkpoint_fname)
    if device.type == "cpu":
        checkpoint = torch.load(checkpoint_fname, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fname)
    model.load_state_dict(checkpoint)
    
    model.eval()
    train_pred = np.ndarray((0,))
    train_measured = np.ndarray((0,))
    for X_train, y_train in train_dataloader:
        predict = model(X_train).reshape(-1)
        pred_y = predict.cpu().detach().numpy().flatten()
        train_measured = np.hstack((train_measured, y_train.cpu().detach().numpy().flatten()))
        train_pred = np.hstack((train_pred, pred_y))
        
    total_mse = 0
    count = 0
    true_all = np.ndarray((0,))
    pred_all = np.ndarray((0,))
    for X_test, y_test in test_dataloader:
        predict = model(X_test).reshape(-1)
        pred_y = predict.cpu().detach().numpy().flatten()
        pred_all = np.hstack((pred_all, pred_y))
        true_all = np.hstack((true_all, y_test.cpu().detach().numpy().flatten()))
        total_mse += mse_sum(predict, y_test).item()
        count += predict.size(0)
    test_loss = math.sqrt(total_mse / count)
    r2 = r2_score(pred_all, true_all)
    
    plt.plot([Min_val, Max_val], [Min_val, Max_val], lw=1, color="black")
    plt.scatter(train_measured, train_pred, s=11, color="blue", alpha=0.8, label="Training")
    plt.scatter(true_all, pred_all, s=11, color="orange", alpha=0.8, label="Testing")
    plt.xlabel("Measured AIMNet2 Binding Energy (kcal/mol)")
    plt.ylabel("Predicted AIMNet2 Binding Energy (kcal/mol)")
    #plt.title(f"Best Test RMSE: {round(test_loss, 2)}, r2: {round(r2, 1)}, EPOCH: {epoch}")
    plt.title(f"{target} - Test RMSE: {round(test_loss, 2)}, r2: {round(r2, 1)}")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    fig.savefig(f"Images/Transformer_{target}.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda")
    with open("Data.json") as jin:
        Data = json.load(jin)
    

    for target in ["7Z0X", "6M0J"]:
        train_dataloader, test_dataloader, Min_val, Max_val = encode_data(Data, target, device)
        
        models = {}
        #models[target], _ = train_transformer_model(device, train_dataloader, test_dataloader, Min_val, Max_val, checkpoint_fname=f"best_{target}.pt")
        transformer_plot(device, train_dataloader, test_dataloader, Min_val, Max_val, checkpoint_fname=f"best_{target}.pt")



