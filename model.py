
import matplotlib.pyplot as plt
import torch
import json
import numpy as np
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from workspace_utils import active_session
from PIL import Image


def training(data_directory,save_directory,arch,learning_rate,hidden_units, epochs, gpu):
    '''
    Split the data into training, testing and validation set
    This function will take the input arguments train the model, display the
    accuracy based on testing set and save the model to checkpoint.pth
    '''
    
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_transforms = transforms.Compose([transforms.RandomRotation(60),transforms.Resize(256),transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data= datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = data_transforms )
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle =False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle =True)
    
    #Device on which the processing takes place
    if gpu:
        device='cuda'
    else:
        device='cpu'
    
    
    class Network(nn.Module):
        def __init__(self,input_size,output_size,hidden_layers,drop_p=0.5):
            super().__init__()
        
            self.hidden_layers =nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
        
            layer_sizes=zip(hidden_layers[:-1],hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_sizes])
            self.output = nn.Linear(hidden_layers[-1],output_size)
        
            self.dropout = nn.Dropout(p=drop_p)
        
        def forward(self, M):
            for linear in self.hidden_layers:
                M = F.relu(linear(M))
                M = self.dropout(M)
            M = self.output(M)
        
            return F.log_softmax(M,dim=1)
    
    def validation(model,valid_loader,criterion):
        validation_loss=0
        validation_accuracy=0
    
        for images,labels in valid_loader:
        
            images,labels=images.to(device),labels.to(device)
            output=model.forward(images)
            validation_loss+=criterion(output,labels).item()
            ps =torch.exp(output)
        
            equality = (labels.data ==ps.max(dim=1)[1])
            validation_accuracy+=equality.type(torch.FloatTensor).mean()
    
        return validation_loss, validation_accuracy    


    pretrained_model = arch
    lr =learning_rate
    drop_p=0.5
        
    torchvision_models= {'vgg16':torchvision.models.vgg16(pretrained = True),
                                            'densenet':torchvision.models.densenet161(pretrained =True),
                        'alexnet':torchvision.models.alexnet(pretrained=True),
                        'resnet152':torchvision.models.resnet152(pretrained=True)}
    model = torchvision_models[pretrained_model]
        
    try:
        input_size= model.classifier[0].in_features
    except TypeError:
        input_size = model.classifier.in_features
    
    output_size = 102
    hidden_layers = hidden_units
        
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = Network(input_size=input_size,output_size=output_size,hidden_layers=hidden_layers,drop_p=drop_p)
        
    model.classifier = classifier
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
        
    model.to(device)
        
    epochs = epochs 
    print_every =40
    steps=0
        
    for e in range(epochs):
        running_loss=0
            
        for ii, (inputs,labels) in enumerate(train_loader):
            steps+=1
                
            images, labels = inputs.to(device), labels.to(device)
                
            optimizer.zero_grad()
                
            outputs = model.forward(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
                
            running_loss+=loss.item()
                
            if steps%print_every ==0:
                model.eval()
            
                with torch.no_grad():
                    validation_loss,validation_accuracy=validation(model, valid_loader, criterion)
            
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_loader)),
                            "Validation Accuracy: {:.3f}".format(100*(validation_accuracy/len(valid_loader))))
        
                running_loss=0
                
                
    print("\n\nMODEL HAS FINISHED TRAINING")
    
    
    #Checking Accuracy of the model
    correct_id = 0
    total_id = 0
    
    with torch.no_grad():
        for data in test_loader: 
        
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _,predicted = torch.max(outputs.data,1)
            total_id += labels.size(0)
            correct_id+=(predicted == labels).sum().item()

    print('\n\nAccuracy of the Network : {} '.format(100*correct_id/total_id))

    #Saving the model to a desired directory
    checkpoint = {'input_size': input_size,
                'output_size' : output_size,
                'hidden_layers':[each.out_features for each in classifier.hidden_layers],
                'class_to_idx':train_data.class_to_idx,
                'state_dict':model.state_dict(),
                 'model' : pretrained_model}
    torch.save(checkpoint,save_directory+'checkpoint.pth')

