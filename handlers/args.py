import argparse

def get_args():
    ar = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ar.add_argument('name', type = str, help = "Please training name - for save folder")
    ar.add_argument('model_name', type = str, help = "Please enter model type - unet,mynet,resnet50")
    ar.add_argument('-p', '--patch_size', type = int, default=150, help = "Please enter model patch size")
    ar.add_argument('-g','--gpu', type = int, default=0, help = "Please enter GPU to use")
    ar.add_argument('-b','--batch_size', type = int, default=32, help = "Please enter model batch size")
    ar.add_argument('-l','--labels', action='append', help="Option to select specific training labels \nOTHER = 0\nTREE = 1\nBUILDING = 2\nCAR = 3\nVEG = 4\nGROUND = 5")
    ar.add_argument('--continue_model', action='store_true', help='Option defining whether to continue existing model')

    dict = {}

    args = ar.parse_args()
    dict.device = 'cuda'+str(args.gpu)
    dict.name = args.name
    dict.model = args.model_name
    dict.patch_size = (args.patch_size,args.patch_size)
    dict.cont = args.continue_model
    label = args.labels
    if label:
        label = list(map(int, label)).sort()
        if not label[0] == 0:
            label = [0]+label
    label.dict = label
    print("Using labels: "+','.join(map(str,label)))
    dict.batch_size = args.batch_size
    return dict