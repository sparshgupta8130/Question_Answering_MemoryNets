
# coding: utf-8

import Skeleton_Code
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specifications of dataset, model and training...')
    parser.add_argument('--folder',action='store', dest='folder_name',
                        type=str,default='en-valid-10k',choices=['en','en-10k','en-valid','en-valid-10k'])
    parser.add_argument('--qa',action='store', dest='qa_name', type=str,default='qa1')
    parser.add_argument('--unk',action='store',dest='unk_thres',type=int,default=0)
    parser.add_argument('--pre_embed',action='store_true')
    parser.add_argument('--m_id',action='store',dest='model_identity',type=str,default='a_model_has_no_name')
    parser.add_argument('--emd',action='store',dest='embedding_dim',type=int,default=10)
    parser.add_argument('--hops',action='store',dest='num_hops',type=int,default=1)
    parser.add_argument('--memsize',action='store',dest='max_mem_size',type=int,default=15)
    parser.add_argument('--epochs',action='store',type=int,default=10)
    parser.add_argument('--eta',action='store',type=float,default=0.0001)
    parser.add_argument('--ls',action='store',dest='LS',type=int,choices=[-1,0,1],default=0)
    parser.add_argument('--ls_thres',action='store',type=float,default=0.001)
    parser.add_argument('--temp',action='store_true')
    parser.add_argument('--posit',action='store_true')
    parser.add_argument('--same',action='store',type=int,choices=[0,1],default=0)
    parser.add_argument('--dropout',action='store',type=float,default=0.0)
    parser.add_argument('--GPU',action='store_true')
    parser.add_argument('--pt2',dest='pyTorch2',action='store_true')
    parser.add_argument('--notest',action='store_false')
    parser.add_argument('--vis',action='store_true')
    args = parser.parse_args()


    Skeleton_Code.train(folder_name=args.folder_name,qa_name=args.qa_name,unk_thres=args.unk_thres,pre_embed=args.pre_embed,
                    model_identity=args.model_identity,embedding_dim=args.embedding_dim,num_hops = args.num_hops,
                    max_mem_size=args.max_mem_size,epochs=args.epochs,eta=args.eta,LS=args.LS,ls_thres=args.ls_thres,
                    temporal=args.temp,positional=args.posit,same=args.same,dropout=args.dropout,visualize=False,
                    GPU=args.GPU,pyTorch2=args.pyTorch2,test=args.notest)

