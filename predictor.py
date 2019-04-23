import pandas as pd
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt
import torch
import json
from Network import Network
import numpy as np
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from workspace_utils import active_session
from PIL import Image

def prediction(Image_path, checkpoint, top_k, category_names, gpu):
    '''
    The function takes the path of the image and the trained model
    topk_k : the top probabilities determined by the model
    category_names : JSON file of the saved category names
    gpu : whether to use GPU. Boolean variable
    '''
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    #Choosing the Processing Device
    if gpu:
        device='cuda'
    else:
        device = 'cpu'
        
    #Rebuilding the model
    
    checkpoint = torch.load(checkpoint)
    torchvision_models= {'vgg16':torchvision.models.vgg16(pretrained = True),
                                            'densenet':torchvision.models.densenet161(pretrained =True),
                        'alexnet':torchvision.models.alexnet(pretrained=True),
                        'resnet152':torchvision.models.resnet152(pretrained=True)}
    model = torchvision_models[checkpoint['model']]
    classifier = Network(checkpoint['input_size'],
                    checkpoint['output_size'], checkpoint['hidden_layers'])
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']


    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
         returns an Numpy array
        '''
    
        new_image = Image.open(image)
    
        #Image transformation 
        image_transform = torchvision.transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        new_image= image_transform(new_image)
    
        #Converting the image to a Numpy array
        new_image = new_image.numpy()
    
        return new_image


    def predict(image_path, trained_model, top_k=5):    
    
        Image = process_image(image_path)
        Image = torch.tensor(Image) #Converting Numpy Image to Tensor
    
        #Adding a dimension to pytorch tensor 
        Image.unsqueeze_(0)
        trained_model = trained_model.to(torch.device(device))
        img = Image.to(torch.device(device))
    
        #Evalutaion mode for prediction
        trained_model.eval()
    
        output = trained_model.forward(img)      
        inv_class_idx = {v: k for k,v in model.class_to_idx.items()}
        probs, inds = torch.topk(output, top_k, sorted=True)   
    
        return probs, inds

    #Printing the probabilities of the predicted image
    
    def sanity_check(path, model):
        probs, classes = predict(path, model)
        probs=probs.detach().cpu().numpy()[0]

        classes=classes.detach().cpu().numpy()[0]
        image = process_image(path)

        max_index = classes[0]

        names = [cat_to_name[str(index)] for index in classes]
        
        probabilities = np.array([n for n in probs])
        
        dataframe = pd.DataFrame(data = pd.Series(data = np.exp(probabilities), index=names), columns = ['Match Probability'])
        
        
        print("\nThe Image is identified as : ", cat_to_name[str(max_index)] )
        print("\n\n THE TOP {} PROBABILITIES".format(top_k))
        
        print(dataframe)
    
    
    sanity_check(Image_path, model)  

