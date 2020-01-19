import argparse

from train_submit import Network

import numpy as np
import pandas as pd

import torch
from torchvision import models
import json
from PIL import Image

def main():
    input_args = get_input_parse()
    
    if input_args.gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print('GPU is not available at the moment! CPU will be used.')
    else:
        device = torch.device('cpu')
        
    image_path = input_args.image_dir
    
    def predict(image_path, model_path, topk=3):
        #Predict the classes of an image using a trained model.
        image_array = process_image(Image.open(image_path))
        image_tensor = torch.from_numpy(image_array).unsqueeze_(0)
        image_tensor = image_tensor.to(device).type(torch.FloatTensor)
        
        model = load_checkpoint(model_path)
        output = model.forward(image_tensor)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk)
        probs, classes = probs.detach().numpy().tolist()[0], classes.detach().numpy().tolist()[0]
        
        inverted_class_to_idx = {val:key for key, val in model.class_to_idx.items()}
        classes = [inverted_class_to_idx[i] for i in classes]
        
        return probs, classes
        
    with open(input_args.cat_name, 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(image_path, input_args.checkpoint, input_args.top_k)
    flower_class = [cat_to_name[i] for i in classes]
    flower_prob = [i*100 for i in probs]
    data = {'Probability(%)': flower_prob, 'Flower Class': flower_class}
    df = pd.DataFrame(data)
    
    print("Most likely: {} ".format(flower_class[0]), "Probabilty: {:.3f}% ".format(probs[0]*100))
    print("\n\nResults Table: \n", df)
    
def get_input_parse():
    #Retrieves and parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='path to the image')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='path to the trained classifier')
    parser.add_argument('--top_k', type=int, default=3, help='set the number of top classes which are most likely to be the flower class')
    parser.add_argument('--cat_name', type=str, default='cat_to_name.json', help='use a .json file to set the label mapping')
    parser.add_argument('--gpu', type=bool, default=True, help='choose changing on a gpu')
    
    return parser.parse_args()
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] is 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    model.classifier = Network(checkpoint['input_size'],
                               checkpoint['output_size'],
                               checkpoint['hidden_layers'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    #Process a PIL image and return a numpy array for the model to predict the class of the image.
    image.thumbnail([256, 256])
    
    left = (image.width - 224) / 2
    right = left + 224
    upper = (image.height - 224) / 2
    lower = upper + 224
    pil_image = image.crop((left, upper, right, lower))
    
    np_image = np.array(pil_image) / 255
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

if __name__ == '__main__':
    main()