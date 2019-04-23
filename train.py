import argparse
from model import training

parser = argparse.ArgumentParser(description = 'Image Classifier Application(Training Mode')

parser.add_argument('data_directory', type=str, default = "",help = 'Directory of the images')
parser.add_argument('--save_directory',type=str,default ="", help= 'Directory to save checkpoint')
parser.add_argument('--arch', type=str, default = 'densenet', help= 'Choose the architecture')
parser.add_argument('--learning_rate', type=int, default=0.001, help='Define a learning rate for the model')
parser.add_argument('--hidden_units',type=list, default= [512,128], help='Define hidden layers. Default layers                              [512,128]')
parser.add_argument('--epochs', type=int, default =5, help = 'Number of times model should perform operations on the network')
parser.add_argument('--gpu', type=bool, default = True, help='Use GPU. Boolean variable (True or False)')

train_args= parser.parse_args()

training(train_args.data_directory,train_args.save_directory,train_args.arch,train_args.learning_rate, train_args.hidden_units,
         train_args.epochs, train_args.gpu)
