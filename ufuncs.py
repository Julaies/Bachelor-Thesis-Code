import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)



def train_loop(data, y, model, criterion, optimizer, lasso=True, factor=1):
    
    total_loss = 0
    size = data.shape[0] * data.shape[1]
    for batch, X in enumerate(data):
        
        # Compute loss
        pred = model(X.unsqueeze(1))
        loss = criterion(pred.squeeze(), y[batch])  # MSE loss
        total_loss +=loss
        # add lasso regression to loss
        if lasso:
            first_params = torch.cat([x.view(-1) for x in model.conv.conv[0].parameters()])
            lasso = factor * torch.norm(first_params, p=1)
            loss += lasso

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print(f"Avg train loss: {total_loss*100/size:>8f}")


def train_loop_reducedlasso(data, y, model, criterion, optimizer, lasso=True, factor1=0.001, factor2 = 0.1):
    '''
    exact same as train_loop, but lasso regularization for the first to filters has factor1 and for the last two filter factor2
    '''
    #data, y = shuffle_batches(data, y)
    total_loss = 0
    #size = data.shape[0] * data.shape[1]
    for batch, X in enumerate(data):
        
        # Compute loss
        pred = model(X.unsqueeze(1))
        loss = criterion(pred.squeeze(), y[batch])  # MSE loss
        total_loss +=loss
        # add lasso regression to loss
        if lasso:
            first_params = torch.cat([x.view(-1) for x in model.conv.conv[0].parameters()])
            lasso = factor1 * torch.norm(first_params[:16] , p=1) + factor2 * torch.norm(first_params[16:] , p=1)
            loss += lasso

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print(f"Avg train loss: {total_loss*100/size:>8f}")



#def test loop:
def test_loop(data, y, model, criterion):
    size = data.shape[0] * data.shape[1]
    test_loss = 0

    with torch.no_grad():
        for i, X in enumerate(data,0):
            pred = model(X.unsqueeze(1)).squeeze()
            test_loss += criterion(pred, y[i]).item()*100   # multiplied by 100 for better readability in output
            #break
    test_loss /= size
  
    print(f"Average test loss:  {test_loss:>8f} \n") 



def predict(data, model, labels):
    length = data.shape[0]
    with torch.no_grad():
        sigmas = np.zeros(length)
        preds = np.zeros(length)
        for i, x in enumerate(data):
            z = model(x.unsqueeze(1)).squeeze(1).cpu().detach().numpy()
            pred = np.argmax(z, axis=1)
            pred = labels[pred]
    
            preds[i] = np.mean(pred, axis=0)
            sigmas[i] = np.std(pred, axis=0)
        return preds, sigmas



#shuffle training data into batches:
def shuffle_data(data, labels, batchsize, shuffle = True): # -> Tuple[torch.tensor, torch.tensor]:

    length, l = data.shape[0], data.shape[1]
    labels = np.repeat(labels, repeats = l, axis = 0)
    data = data.reshape(l*length, 32, 32)

    if shuffle: 
        p = np.random.permutation(l * length)
        data, labels = data[p], labels[p]

    n_batches = int(l * length / batchsize)
    data = torch.from_numpy(data).to(torch_device).view(n_batches, batchsize,32,32)
    labels = torch.from_numpy(labels).to(torch_device).view(n_batches, batchsize,length)
    return data.float(), labels.float()



def shuffle_data_3d(data, labels, batchsize, shuffle = True): # -> Tuple[torch.tensor, torch.tensor]:

    length, l = data.shape[0], data.shape[1]
    labels = np.repeat(labels, repeats = l, axis = 0)
    data = data.reshape(l*length, 16, 16, 16)

    if shuffle: 
        p = np.random.permutation(l * length)
        data, labels = data[p], labels[p]

    n_batches = int(l * length / batchsize)
    data = torch.from_numpy(data).to(torch_device).view(n_batches, batchsize,16,16,16)
    labels = torch.from_numpy(labels).to(torch_device).view(n_batches, batchsize,length)
    return data.float(), labels.float()