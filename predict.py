import argparse
from predictor import prediction

parser = argparse.ArgumentParser(description = 'Image Classifier Application(Test Mode)')

parser.add_argument('image_path', type=str, help="Path of the image file")
parser.add_argument('checkpoint',type=str, help="Path to checkpoint file")
parser.add_argument('--top_k', type = int, default = 5 ,help="Value of top_k")
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Category name file')
parser.add_argument('--gpu', type=bool, default = True, help= "Use GPU, Boolean Variable(True or False)")

predict_args = parser.parse_args()

prediction(predict_args.image_path, predict_args.checkpoint, predict_args.top_k,
          predict_args.category_names,predict_args.gpu)


                    