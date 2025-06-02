#!/usr/bin/env python
import torch
import argparse
import multiprocessing as mp

from agent.training import Training

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.') #创建命令行参数  ### argparse 是 Python 标准库中用于处理命令行参数的模块，能帮你把：python train.py --learning_rate 0.001 --restore
                                                                         #这样的命令行输入，自动转换成一个 Python 字典结构：args['learning_rate'] = 0.001 args['restore'] = True
                                                                        #description='Deep reactive agent.' 是这个程序的描述，会在你运行 python xxx.py --help 时显示出来，告诉你这个程序是干什么的。
    parser.add_argument('--entropy_beta', type=float, default=0.01,   #parser.add_argument('--xxx', ...) 就是告诉程序：我要从命令行接收一个叫 --xxx 的参数。
                        help='entropy beta (default: 0.01)')          #help="..."

    parser.add_argument('--restore', action='store_true', help='restore from checkpoint')  #决定是否从 checkpoint 加载模型继续训练 #用 action='store_true'，意味着：如果你写 --restore，值是 True
    #Checkpoint 就是“模型的保存文件”，在训练神经网络的时候，我们会定期把模型的参数保存下来，比如每训练 1 万步就保存一次，生成一个文件：checkpoint-10000.pth checkpoint-20000.pth ...
    parser.add_argument('--grad_norm', type = float, default=40.0,     #限制梯度大小，防止梯度爆炸（梯度裁剪）
        help='gradient norm clip (default: 40.0)')

    parser.add_argument('--h5_file_path', type = str, default='/app/data/{scene}.h5')   #设置数据集 .h5 文件路径（存储了场景信息或图像）
    parser.add_argument('--checkpoint_path', type = str, default='/model/checkpoint-{checkpoint}.pth') #设置模型参数保存/加载的路径

    parser.add_argument('--learning_rate', type = float, default= 0.0007001643593729748) #学习率
    parser.add_argument('--rmsp_alpha', type = float, default = 0.99,   #RMSProp 优化器的衰减系数（类似动量）
        help='decay parameter for RMSProp optimizer (default: 0.99)')
    parser.add_argument('--rmsp_epsilon', type = float, default = 0.1,  #RMSProp 的稳定项 ε，防止除以 0，增加数值稳定性
        help='epsilon parameter for RMSProp optimizer (default: 0.1)')


    
 #parser.parse_args() 这个是 argparse 的标准用法，它会解析命令行参数，返回一个 Namespace 对象
#比如命令行这样运行：python train.py --learning_rate 0.001 --restore
#那么args = parser.parse_args()会返回 Namespace(learning_rate=0.001, restore=True)
    
    args = vars(parser.parse_args())   #vars() 是 Python 的内置函数，作用是：把对象的属性转换为字典。

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #如果你想使用 GPU，可以把注释的那行放开
    device = torch.device('cpu')

    if args['restore']:
        t = Training.load_checkpoint(args)
    else:
        t = Training(device, args)    #新建一个 Training 类的实例，从头训练

    t.run()



