import argparse
import os
def getparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', help='path to dataset', default='data/')
    parser.add_argument('--name', help='network name', default='')
    parser.add_argument('--model', default='dcgan', help='Type of model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=5000, help='num epochs')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--cuda', action='store_false', help='disables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--inf', default='', help="path to created models")
    parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    parser.add_argument('--update_iter', type=int, help='Number of Discriminator updates before Generator update',
                        default=1)
    args = parser.parse_args()
    try:
        os.makedirs(args.outf)
    except OSError:
        print("error")
    if args.name =='':
        args.name = args.model +"_"+str(args.batch_size)+"_"+str(args.nz)
    args.outf = args.outf+"/"+args.name
    try:
        os.makedirs(args.outf)
    except OSError:
        print("error")
    return args