import argparse

import torch
from torch import optim, nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms

def main():
    input_args = get_input_args()
    
    if input_args.gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print('GPU is not available at the moment! CPU will be used.')
    else:
        device = torch.device('cpu')
    
    train_dir = input_args.data_dir
    valid_dir = 'flowers/valid'
    test_dir = 'flowers/test'
    
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(45),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])}
    
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}
    
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)}
    
    model = calling_model(input_args.arch)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = Network(25088, 102, input_args.hid_un, drop_p=0.5)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=input_args.lr)
    
    model.to(device)
    
    def validation(model, dataloaders_valid, criterion):
        valid_loss = 0
        accuracy = 0
        for images, labels in dataloaders_valid:
            
            images, labels = images.to(device), labels.to(device)
            
            output = model.forward(images)
            valid_loss += criterion(output, labels).item()
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
            
        return valid_loss, accuracy
    
    epochs = input_args.epochs
    print_every = 40
    steps = 0
    running_loss = 0
    
    for e in range(epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, dataloaders['valid'], criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(dataloaders['valid'])),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))
                
                running_loss = 0
                
                model.train()
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'arch': input_args.arch,
                  'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'state_dict': model.classifier.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict()}
    
    torch.save(checkpoint, input_args.save_dir)

def get_input_args():
    #Retrieves and parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default='flowers/train', help='path to folder of images for training')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='set directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg19'], help='choose the architecture')
    parser.add_argument('--lr', type=float, default=0.001, help='set the learning rate')
    parser.add_argument('--hid_un', type=list, default=[5000, 500], help='set the number of hidden units(format: [layer1, layer2])')
    parser.add_argument('--epochs', type=int, default=8, help='set the number of training epochs')
    parser.add_argument('--gpu', type=bool, default=True, help='choose training on a GPU')
    
    return parser.parse_args()

def calling_model(model='vgg16'):
    #Choose the model that will be used for training.
    if model is 'vgg16':
        chosen_model = models.vgg16(pretrained=True)
    elif model is 'vgg19':
        chosen_model = models.vgg19(pretrained=True)
    
    return chosen_model

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    main()