import argparse
import sys
import os

def get_args():
    ar = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ar.add_argument('-n', '--name', type = str, default =  "test", help = "Please training name - for save folder")
    ar.add_argument('-m', '--model', type = str, default =  "mynet",help = "Please enter model type - unet,mynet,resnet50")
    ar.add_argument('-i', '--input_folder', type = str, default="../input/", help = "Please enter image location")
    ar.add_argument('-o', '--output_folder', type = str, default="../results/", help = "Please enter model save location")
    ar.add_argument('-p', '--patch_size', type = int, default=128, help = "Please enter model patch size")
    ar.add_argument('-g','--gpu', type = int, default=0, help = "Please enter GPU to use")
    ar.add_argument('-b','--batch_size', type = int, default=64, help = "Please enter model batch size")
    ar.add_argument('-l','--labels', action='append', help="Option to select specific training labels \nOTHER = 0\nTREE = 1\nBUILDING = 2\nCAR = 3\nVEG = 4\nGROUND = 5")
    ar.add_argument('--cont', action='store_true', help='Option defining whether to continue existing model')

    args = ar.parse_args()
    args.patch_size = (args.patch_size,args.patch_size)
    label = args.labels
    if label:
        label = list(map(int, label)).sort()
        if not label[0] == 0:
            label = [0]+label
        print("Using labels: "+','.join(map(str,label)))    
    args.label = label

    #----Train Result Paths---
    args.output_folder += args.name
    if os.path.exists(args.output_folder):
        if not args.cont:
            print("Training path already exists")
            #sys.exit(0)
    else:
        if args.cont:
            print("Model does not exist")
            sys.exit(0)
        os.makedirs(args.output_folder)

    return args