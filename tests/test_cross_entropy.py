import torch.nn as nn
import torch

loss = nn.CrossEntropyLoss()
# input_ = torch.tensor([[ 1.0382, -0.3135,  0.0972,  1.4217, -1.2889],
#         [-1.2384, -1.6011, -0.1489,  0.4537, -0.0805],
#         [ 0.9158,  0.5404, -0.9637, -1.3458,  0.8143]]).to(torch.float32)

input_ = torch.tensor([[0, 0, 0, 100, 0], [0, 0, 0, 100, 0], [100, 0, 0, 0, 0]]).to(torch.float32)
print(input_)
target = torch.tensor([3, 3, 0]).to(torch.int64)
print(target)
output = loss(input_, target)
print('loss: ', output.item())

print(input_.T.shape)

# losses = 0
# for i in range(5):
#     new_tensor = input_[:, i].T.unsqueeze(1)
#     losses += loss(new_tensor, target).item()


# print('summed losses: ', losses)

# Example of target with class probabilities
    # input_ = torch.randn(3, 5, requires_grad=True)
    # print(input_)
    # target = torch.randn(3, 5).softmax(dim=1)
    # print(target)
    # output = loss(input_, target)
    # print('loss: ', output.item())