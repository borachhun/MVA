from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
from Model import Discriminator
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd

path = '/Data/.Data/ALTEGRAD'
cache_directory = "/Data/.cache/HuggingFace" 

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

model_name = 'distilbert-base-uncased'
# model_name = "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
gt = np.load(path+"/data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root=path+'/data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root=path+'/data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 60
batch_size = 32
learning_rate = 2e-5

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300\
              , graph_hidden_channels=300, cache_dir=cache_directory) # nout = bert model hidden dim
model.to(device)

save_path = './model/model_update.pt'

if os.path.exists(save_path):
    checkpoint = torch.load(save_path, map_location=device)  # 确保加载到正确的设备
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded model from {save_path}")
else:
    print(f"No saved model found at {save_path}")

D_s = Discriminator(input_dim=768, feature_dim=128, num_residual_layers=2)
D_s.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

optimizer_d = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

epoch = 0
loss = 0
loss_w = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

one = torch.FloatTensor([1]).to(device)
mone = one * -1

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for batch in train_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))

        '''
        Train Discriminator
        '''
        # Generator update
        for p in D_s.parameters():
            p.requires_grad = True  # to avoid computation
        for p in model.parameters():
            p.requires_grad = False  # to avoid computation

        optimizer_d.zero_grad()
        
        d_loss_real = D_s(x_text)
        d_loss_real = d_loss_real.mean(0).view(1)
        d_loss_real.backward(gradient=one, retain_graph=True)

        d_loss_fake = D_s(x_graph)
        d_loss_fake = d_loss_fake.mean(0).view(1)
        d_loss_fake.backward(gradient=mone, retain_graph=True)

        d_loss = d_loss_fake - d_loss_real
        Wasserstein_D = d_loss_real - d_loss_fake
        optimizer_d.step()
        
        '''
        Train Model (Txt and Graph Encoders)
        '''
        for p in D_s.parameters():
            p.requires_grad = False  # to avoid computation
        for p in model.parameters():
            p.requires_grad = True  # to avoid computation

        optimizer.zero_grad()
        
        current_loss = contrastive_loss(x_graph, x_text) # ONLY FOR PRINGTING
        current_loss.backward(retain_graph=True)
        
        model_loss = D_s(x_graph) # Use Wsst Distance
        model_loss = model_loss.mean().mean(0).view(1)
        model_loss.backward(gradient=one, retain_graph=True)

        model_t_loss = D_s(x_text) # Use Wsst Distance
        model_t_loss = model_loss.mean().mean(0).view(1)
        model_t_loss.backward(gradient=mone, retain_graph=True)
        
        optimizer.step()
        
        loss += current_loss.item()
        loss_w += model_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery),
                                                                        f'loss_w: {loss_w/printEvery}')
            losses.append(loss)
            loss = 0 
    model.eval()       
    val_loss = 0        
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    if best_validation_loss==val_loss:
        print('validation loss improoved saving checkpoint...')
        # save_path = os.path.join('./model', 'model'+str(i)+'.pt')
        
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))

np.save('losses_array.npy', np.array(losses))

print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root=path+'/data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path=path+'/data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)

text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)