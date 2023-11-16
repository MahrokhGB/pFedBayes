import torch
from tqdm import tqdm

# from FLAlgorithms.users.userpFedbayes import UserpFedBayes
# from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np


# Implementation for FedAvg Server
# class pFedBayes(Server):
#     def __init__(self, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
#                  local_epochs, optimizer, num_users, times, device, personal_learning_rate,
#                  output_dim=10, post_fix_str=''):
#         super().__init__(dataset, algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
#                          local_epochs, optimizer, num_users, times, device)

dataset = 'femnist_reduced' # 'Mnist'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize data for all  users
data = read_data(dataset)
print(data[0])
print('type(data)', type(data)) # tuple
print('len(data)', len(data))   # 4
# total_users = len(data[0])
total_users = 1
print('clients initializting...')
for i in tqdm(range(total_users), total=total_users):
    id, train, test = read_user_data(i, data, dataset, device)
    print('id', id) # 'f_00000', ..., 'f_00009'
    print(type(train)) # list
    print(len(train), len(train)) # 1000
    print(type(test))
    print('len(test)', len(test)) #4000

# print(type(train[0])) 'tuple'
# print(len(train[0])) 2
print(type(train[0][0])) # torch.Tensor
print(train[0][0].shape) # torch.Size([1, 28, 28])
print(type(train[0][1])) # torch.Tensor
print(train[0][1])       # scalar tensor



    # user = UserpFedBayes(id, train, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer,
    #                         personal_learning_rate, device, output_dim=output_dim)
    # self.users.append(user)
    # self.total_train_samples += user.train_samples
