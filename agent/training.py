from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.training_thread import TrainingThread
from agent.optim import SharedRMSprop
from typing import Collection, List
import torch.nn as nn
import torch.multiprocessing as mp 
import logging
import sys
import torch
import os
import threading
from contextlib import suppress
import re

TOTAL_PROCESSED_FRAMES = 20 * 10**6 # 10 million frames    #这表示训练过程中，模型会处理 2000 万帧图像
TASK_LIST = {
  'bathroom_02': ['26', '37', '43', '53', '69'],
  'bedroom_04': ['134', '264', '320', '384', '387'],
  'kitchen_02': ['90', '136', '157', '207', '329'],
  'living_room_08': ['92', '135', '193', '228', '254']
}

DEFAULT_CONFIG = {
    'saving_period': 10 ** 6 // 5,
    'checkpoint_path': 'model/checkpoint-{checkpoint}.pth',
    'grad_norm': 40.0,
    'gamma': 0.99,
    'entropy_beta': 0.01,
    'max_t': 5,
}

class TrainingSaver:
    def __init__(self, shared_network, scene_networks, optimizer, config):  #这里的self就是TrainingSaver本身  ## 当你创建一个类的实例时（也就是“对象”），Python 会自动调用这个 __init__ 方法来初始化这个对象的属性
        self.config = config     #self.my_config = config  这样写也完全没问题，self.my_config 就是你给对象取的属性名，config 是你传进来的参数名。名字你定，规则是赋值。
        n_config = DEFAULT_CONFIG.copy() #	.copy() 是字典的方法，表示复制一份新字典出来，防止修改原始的 DEFAULT_CONFIG
        n_config.update(config)        # .update() 会将 config 中的键值对加到 n_config 中，如果有重复的 key，会覆盖默认值。 这样子n_config就是一个新的config,输入的参数更新，没输入的参数保持默认值
        self.config.update(n_config)  
        self.checkpoint_path = self.config['checkpoint_path'] #	这里从配置字典里取出键 'checkpoint_path' 对应的值，赋给对象的属性 self.checkpoint_path  这样做的目的是方便直接通过 self.checkpoint_path 使用，不用每次都写 self.config['checkpoint_path']。
        self.saving_period = self.config['saving_period']   
        self.shared_network = shared_network
        self.scene_networks = scene_networks
        self.optimizer = optimizer        

  #在优化器执行完一次优化之后，判断是否该保存模型（checkpoint）。
    def after_optimization(self):
        iteration = self.optimizer.get_global_step() #调用优化器的 get_global_step() 方法，得到当前已经处理了多少步（step）或帧（frame）
        if iteration % self.saving_period == 0:   #self.saving_period 是一个保存周期，比如设置为 2_000_000。
            self.save()   #保存下来


  #打印出模型训练所用的配置参数，可以帮助你检查当前配置设置
    def print_config(self, offset: int = 0):  #offset: int = 0 是函数参数，表示每行前面要缩进多少个空格，默认是 0。
        for key, val in self.config.items():  #遍历配置字典 self.config 中的所有键值对
            print((" " * offset) + f"{key}: {val}") #	" " * offset 会生成一定数量的空格（用于缩进） f"{key}: {val}" 是 f-string，用于格式化输出。
        pass #pass 是占位符，表示什么都不做。这行其实可以删掉，不影响程序。

    def save(self):
        iteration = self.optimizer.get_global_step() #获取当前已经训练了多少步（frames）
        filename = self.checkpoint_path.replace('{checkpoint}', str(iteration)) #用实际的步数（如 1000000）替换掉路径模板中的 {checkpoint} 占位符。 例如：'model/checkpoint-{checkpoint}.pth' → 'model/checkpoint-1000000.pth'
        model = dict() #创建一个空的 model 字典，用于打包所有要保存的信息
        model['navigation'] = self.shared_network.state_dict() #self.shared_network.state_dict() 会返回神经网络的所有权重参数
        for key, val in self.scene_networks.items(): #遍历每个场景对应的子网络（如 bathroom、kitchen 等）。  # 字典对象 才有 .items() 方法，作用是返回所有的键值对（key-value pairs）：
            model[f'navigation/{key}'] = val.state_dict() #把它们的参数也加到 model 字典里，key 形式如：navigation/bathroom_02。
        model['optimizer'] = self.optimizer.state_dict() #保存优化器的状态（如学习率、动量、历史梯度等），这样下次恢复时训练可以无缝继续
        model['config'] = self.config #把当前的配置参数也保存进去，方便以后对照或重现实验。
        
        with suppress(FileExistsError): #如果保存目录不存在，就自动创建。用 suppress(FileExistsError) 是为了防止并发写入时报错
            os.makedirs(os.path.dirname(filename)) 
        torch.save(model, open(filename, 'wb')) #使用 torch.save() 把 model 字典写入磁盘，保存为二进制文件（.pth）  wb 表示以二进制写入模式打开文件

    def restore(self, state): 
        if 'optimizer' in state and self.optimizer is not None: self.optimizer.load_state_dict(state['optimizer'])   #如果 state 字典中有 'optimizer' 键，并且当前对象 self.optimizer 不是空的，就把保存的优化器状态加载回来，这样训练就可以从之前的位置接着优化
        if 'config' in state: #如果保存的模型里包含配置信息 'config'
            n_config = state['config'].copy() #先复制出一份 state['config']
            n_config.update(self.config)  #然后把当前实例的 config 覆盖进去（优先使用当前用户输入的配置）
            self.config.update(n_config) #最终把这个新的配置 n_config 更新进 self.config

        self.shared_network.load_state_dict(state['navigation']) #恢复共享网络的权重

        tasks = self.config.get('tasks', TASK_LIST) #从字典 self.config 中尝试获取 'tasks' 这个键对应的值，如果 找不到，就使用默认的 TASK_LIST
        for scene in tasks.keys(): #对于每一个场景（如 kitchen_02, bedroom_04 等）
            self.scene_networks[scene].load_state_dict(state[f'navigation/{scene}']) #加载对应子网络的参数

class TrainingOptimizer:
    def __init__(self, grad_norm, optimizer, scheduler):
        self.optimizer : torch.optim.Optimizer = optimizer #第一部分：self.optimizer = optimizer 把传进来的 optimizer（一个优化器对象）赋值给当前对象的属性 self.optimizer 
                                                         # 第二部分：: torch.optim.Optimizer 这就是 类型注解，意思是告诉你（和 IDE、静态分析工具）：self.optimizer 这个变量的类型应该是 torch.optim.Optimizer
                                                         #类似用法：x: int = 10 ### name: str = "hello"
        self.scheduler = scheduler #保存学习率调度器。这个对象会在训练中定期调整学习率。
        self.grad_norm = grad_norm #保存梯度裁剪的阈值（比如 40.0），训练时用于防止“梯度爆炸”。
        self.global_step = torch.tensor(0) #初始化一个训练的全局步数计数器，类型为 PyTorch 的张量（torch.tensor），可以用于记录当前优化了多少步
        self.lock = mp.Lock() # mp 是 Python 的多进程模块 multiprocessing。这个锁是为了在多线程或多进程训练中，保护对共享资源（如 global_step、模型参数）的访问，防止数据冲突。

    def state_dict(self):
        state_dict = dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict["global_step"] = self.global_step
        return state_dict

    def share_memory(self):
        self.global_step.share_memory_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.global_step.copy_(state_dict['global_step'])
    
    def get_global_step(self):
        return self.global_step.item()

        
    def _ensure_shared_grads(self, local_params, shared_params):
        for param, shared_param in zip(local_params, shared_params):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def optimize(self, loss, local_params, shared_params):
        local_params = list(local_params)
        shared_params = list(shared_params)

        # Fix the optimizer property after unpickling
        self.scheduler.optimizer = self.optimizer
        self.scheduler.step(self.global_step.item())

        # Increment step
        with self.lock:
            self.global_step.copy_(torch.tensor(self.global_step.item() + 1))
            
        self.optimizer.zero_grad()

        # Calculate the new gradient with the respect to the local network
        loss.backward()

        # Clip gradient
        torch.nn.utils.clip_grad_norm_(local_params, self.grad_norm)
            
        self._ensure_shared_grads(local_params, shared_params)
        self.optimizer.step()

class AnnealingLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        super(AnnealingLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0 - self.last_epoch / self.total_epochs)
                for base_lr in self.base_lrs]

class Training:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.logger : logging.Logger = self._init_logger()
        self.learning_rate = config.get('learning_rate')
        self.rmsp_alpha = config.get('rmsp_alpha')
        self.rmsp_epsilon = config.get('rmsp_epsilon')
        self.grad_norm = config.get('grad_norm', 40.0)
        self.tasks = config.get('tasks', TASK_LIST)
        self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        self.max_t = config.get('max_t', 5)
        self.total_epochs = TOTAL_PROCESSED_FRAMES // self.max_t
        self.initialize()

    @staticmethod
    def load_checkpoint(config, fail = True):
        device = torch.device('cpu')
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        max_t = config.get('max_t', 5)
        total_epochs = TOTAL_PROCESSED_FRAMES // max_t
        files = os.listdir(os.path.dirname(checkpoint_path))
        base_name = os.path.basename(checkpoint_path)
        
        # Find latest checkpoint
        # TODO: improve speed
        restore_point = None
        if base_name.find('{checkpoint}') != -1:
            regex = re.escape(base_name).replace(re.escape('{checkpoint}'), '(\d+)')
            points = [(fname, int(match.group(1))) for (fname, match) in ((fname, re.match(regex, fname),) for fname in files) if not match is None]
            if len(points) == 0:
                if fail:
                    raise Exception('Restore point not found')
                else: return None
            
            (base_name, restore_point) = max(points, key = lambda x: x[1])

            
        print(f'Restoring from checkpoint {restore_point}')
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        training = Training(device, state['config'] if 'config' in state else config)
        training.saver.restore(state) 
        print('Configuration')
        training.saver.print_config(offset = 4)       
        return training

    def initialize(self):
        # Shared network
        self.shared_network = SharedNetwork()
        self.scene_networks = { key:SceneSpecificNetwork(4) for key in TASK_LIST.keys() }

        # Share memory
        self.shared_network.share_memory()
        for net in self.scene_networks.values():
            net.share_memory()

        # Callect all parameters from all networks
        parameters = list(self.shared_network.parameters())
        for net in self.scene_networks.values():
            parameters.extend(net.parameters())

        # Create optimizer
        optimizer = SharedRMSprop(parameters, eps=self.rmsp_epsilon, alpha=self.rmsp_alpha, lr=self.learning_rate)
        optimizer.share_memory()

        # Create scheduler
        scheduler = AnnealingLRScheduler(optimizer, self.total_epochs)

        # Create optimizer wrapper
        optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, scheduler)
        self.optimizer = optimizer_wrapper
        optimizer_wrapper.share_memory()

        # Initialize saver
        self.saver = TrainingSaver(self.shared_network, self.scene_networks, self.optimizer, self.config)
    
    def run(self):
        self.logger.info("Training started")

        # Prepare threads
        branches = [(scene, int(target)) for scene in TASK_LIST.keys() for target in TASK_LIST.get(scene)]
        def _createThread(id, task):
            (scene, target) = task
            net = nn.Sequential(self.shared_network, self.scene_networks[scene])
            net.share_memory()
            return TrainingThread(
                id = id,
                optimizer = self.optimizer,
                network = net,
                scene = scene,
                saver = self.saver,
                max_t = self.max_t,
                terminal_state_id = target,
                **self.config)

        self.threads = [_createThread(i, task) for i, task in enumerate(branches)]
        
        try:
            for thread in self.threads:
                thread.start()

            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            # we will save the training
            print('Saving training session')
            self.saver.save()
        

    def _init_logger(self):
        logger = logging.getLogger('agent')
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger
