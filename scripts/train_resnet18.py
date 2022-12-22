import sys
if '..' not in sys.path:
    sys.path.append('..')
    
if '.' not in sys.path:
    sys.path.append('.')
    
print(sys.path)
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet

from source.models import BasicBlock
from source.data import get_training_dataloader, get_test_dataloader


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MILESTONES = [60, 120, 160]
# data_path = '/gpfs/gpfs0/k.sobolev/cifar-100-python/cifar-100-python/'
data_path = '../data'


cifar100_training_loader = get_training_dataloader(
    data_path,
    CIFAR100_TRAIN_MEAN,
    CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=64,
    shuffle=True
)

cifar100_test_loader = get_test_dataloader(
    data_path,
    CIFAR100_TRAIN_MEAN,
    CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=64,
    shuffle=True
)

model = ResNet(num_classes=100, block=BasicBlock, layers=[2, 2, 2, 2])
model.eval()
model = model.cuda()

loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

best_acc = 0.0
for epoch in range(200):
    model.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        labels = labels.cuda()
        images = images.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0.0
        
        for (images, labels) in cifar100_test_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            val_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
    
    val_acc = correct.float() / len(cifar100_test_loader.dataset)
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        val_loss / len(cifar100_test_loader.dataset),
        val_acc,
    ))
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'models/resnet18_cifar100.sd')
