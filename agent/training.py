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

    def state_dict(self): #这个 state_dict() 方法的作用是 将当前优化器、调度器、训练步数打包成一个字典，用于保存或恢复训练状态。
        state_dict = dict() #创建一个空的字典
        state_dict['optimizer'] = self.optimizer.state_dict() #把它的参数（如每一层的动量、学习率、自适应项等）保存下来
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict["global_step"] = self.global_step
        return state_dict

    def share_memory(self):
        self.global_step.share_memory_() #是用来 将 global_step 放入共享内存（shared memory）中，使得多个进程（processes）之间都可以访问和修改它，而不是各自复制一份。它通常用于 多进程训练 中的进度同步

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer']) #从 state_dict 中取出保存的优化器状态（比如动量、历史梯度等），加载回当前的 optimizer 对象中。
        self.scheduler.load_state_dict(state_dict['scheduler']) 
        self.global_step.copy_(state_dict['global_step'])
    
    def get_global_step(self):
        return self.global_step.item() #.item() 方法是 PyTorch 中的函数，用来从只包含一个元素的张量中提取出 Python 标量值（如 int 或 float）

        
    def _ensure_shared_grads(self, local_params, shared_params): #一个私有方法（以下划线开头通常表示“仅供内部使用”）
        for param, shared_param in zip(local_params, shared_params): #zip() 同时迭代两个参数列表中的每对参数
            if shared_param.grad is not None: #如果共享模型的这个参数已经有梯度了，就直接退出整个函数，不再处理其他参数。 这通常表示这个共享模型的梯度已经被其他进程同步过了，不需要重复。
                return
            shared_param._grad = param.grad #如果共享模型的参数还没有梯度，就把本地模型的梯度复制过去。
            #注意使用的是 _grad（带下划线），这是 PyTorch 中的一个 内部属性，允许你手动设置梯度。
            #PyTorch 默认情况下不能跨进程自动同步梯度，所以必须 手动赋值梯度，这正是这个函数的作用。

  #为什么要这么做？
  #在使用多进程训练（如 A3C）时，每个进程都有一份“本地模型”，但训练时会：
  #使用 本地模型与环境交互
  #将 本地计算出的梯度同步到共享模型
  #最终用共享模型参数做优化（即参数更新）

    def optimize(self, loss, local_params, shared_params):  
        local_params = list(local_params)
        shared_params = list(shared_params)

        # Fix the optimizer property after unpickling
        self.scheduler.optimizer = self.optimizer #scheduler 是一个对象，它本身内部就有一个属性叫 optimizer，代表它控制的是哪个优化器。
        self.scheduler.step(self.global_step.item()) #scheduler.step(step_number) 是在告诉调度器：“我现在已经训练到第 step_number 步了，你根据这个步数来决定当前学习率是多少。”
      #有些学习率调度器是根据训练的步数（或 epoch）来决定学习率的：scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.95 ** step)
      #那么每走一步，学习率就变成原来的 0.95 倍。
      #所以上面这句话的意思是“我现在训练到了第 N 步，帮我调整学习率。”
      
      #什么是scheduler
        #scheduler 全称叫 学习率调度器（learning rate scheduler），是 PyTorch 中用于 自动调整学习率 的模块。
        #它的作用：在训练过程中，动态地改变学习率，从而加快收敛、避免震荡、提升最终精度。
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 这表示：每训练 10 个 epoch，把学习率缩小为原来的 0.1 倍。



        # Increment step
        with self.lock:  #使用多进程锁 self.lock，避免多个进程同时写 global_step。
            self.global_step.copy_(torch.tensor(self.global_step.item() + 1)) #读取当前 global_step，加一，再写回去。
            
        self.optimizer.zero_grad() #清除旧的梯度，防止累积。

        # Calculate the new gradient with the respect to the local network
        loss.backward() #反向传播，计算梯度

        # Clip gradient
        torch.nn.utils.clip_grad_norm_(local_params, self.grad_norm) #	进行梯度裁剪（clip gradient），防止梯度爆炸。 	# 如果梯度 L2 范数超过 self.grad_norm（比如 40.0），就缩放下来
            
        self._ensure_shared_grads(local_params, shared_params) #把本地参数的 .grad 复制到共享参数 .grad 上。
        self.optimizer.step() #用共享模型的梯度更新参数，完成梯度下降。

#1. 准备参数列表
#2. 调整调度器
#3. 用锁同步 global_step++
#4. 反向传播得到 local 梯度
#5. 梯度裁剪（clip）
#6. 把 local 梯度传给 shared 参数
#7. 优化器 step，更新 shared 参数

class AnnealingLRScheduler(torch.optim.lr_scheduler._LRScheduler): #定义了一个新类，继承自 PyTorch 的 “_LRScheduler”。这是 PyTorch 中所有学习率调度器的父类。 #目的就是实现学习率衰减的功能，因为原来的父类“_LRScheduler”，没有学习率衰减的功能
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
      #这里初始化self.optimizer,self.last_epoch,self.total_epochs就是为了下面的get_lf函数做准备的
        self.optimizer = optimizer #调度其学习率的优化器（如 SGD、Adam）
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        super(AnnealingLRScheduler, self).__init__(optimizer, last_epoch) #调用 父类 _LRScheduler 的初始化方法。
                                                                          #为什么要传 optimizer, last_epoch？因为 PyTorch 的 _LRScheduler 的构造函数是这样的：def __init__(self, optimizer, last_epoch=-1):
    def get_lr(self):
        return [base_lr * (1.0 - self.last_epoch / self.total_epochs)   #计算当前 epoch（训练轮次）下的学习率，用的是一种简单的线性衰减策略（linear annealing）
                for base_lr in self.base_lrs]    #self.base_lrs: 是一个列表，保存了每个参数组的初始学习率（由 optimizer 提供）
                            #self.last_epoch: 当前是第几个 epoch。每次 scheduler.step() 被调用时，它会加 1

class Training:
    def __init__(self, device, config): #从配置中读取各种训练参数，并进行初始化准备
        self.device = device
        self.config = config
        self.logger : logging.Logger = self._init_logger()   #self._init_logger() 是一个私有方法，用来初始化日志记录器（写日志到控制台或文件，方便调试和记录训练过程）。
        self.learning_rate = config.get('learning_rate')
        self.rmsp_alpha = config.get('rmsp_alpha') #rmsp_alpha: RMSprop 优化器的 alpha 参数（控制平方梯度的平均）。
        self.rmsp_epsilon = config.get('rmsp_epsilon') #rmsp_alpha: RMSprop 优化器的 alpha 参数（控制平方梯度的平均）。
        self.grad_norm = config.get('grad_norm', 40.0) 
        self.tasks = config.get('tasks', TASK_LIST)
        self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        self.max_t = config.get('max_t', 5)  #每次 rollout 的最大步数（动作的时间步长度），默认是 5。
        self.total_epochs = TOTAL_PROCESSED_FRAMES // self.max_t
        self.initialize()

    @staticmethod   #@staticmethod：说明这个方法不依赖类的实例（self），可以通过 Training.load_checkpoint(...) 直接调用
    def load_checkpoint(config, fail = True):  #	fail=True：是否在找不到 checkpoint 时报错（True 表示报错；False 表示返回 None）
        device = torch.device('cpu')
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth') 
        max_t = config.get('max_t', 5)
        total_epochs = TOTAL_PROCESSED_FRAMES // max_t
        files = os.listdir(os.path.dirname(checkpoint_path)) #获取 checkpoint 目录下的所有文件名
        base_name = os.path.basename(checkpoint_path) #获取文件模板名，如 checkpoint-{checkpoint}.pth
        
        # Find latest checkpoint
        # TODO: improve speed
        restore_point = None
        if base_name.find('{checkpoint}') != -1: #判断 base_name 这个字符串中是否包含子字符串 "{checkpoint}"。如果找到，返回它的起始位置；如果找不到，返回 -1
            regex = re.escape(base_name).replace(re.escape('{checkpoint}'), '(\d+)')   #re.escape('{checkpoint}')  → '\\{checkpoint\\}'   #regex = 'checkpoint\\-(\\d+)\\.pth'
            points = [(fname, int(match.group(1))) for (fname, match) in ((fname, re.match(regex, fname),) for fname in files) if not match is None] #在给定目录下的所有文件中，找出文件名匹配 regex（比如 checkpoint-xxxxx.pth）的那些，并提取出其中的数字（即 checkpoint 的编号）。
            if len(points) == 0:
                if fail:
                    raise Exception('Restore point not found')
                else: return None
            
            (base_name, restore_point) = max(points, key = lambda x: x[1])   #找出 checkpoint 数字最大的那个文件，即“最新的模型”

            
        print(f'Restoring from checkpoint {restore_point}') #这行是打印信息，告诉你当前从哪个 checkpoint 文件（如 checkpoint-200000.pth）中恢复。
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
      #os.path.dirname(checkpoint_path)：获取路径中不含文件名的部分（即目录名） 
      # os.path.join(...)：拼接目录路径和文件名 base_name，得到完整路径
     # open(..., 'rb')：以二进制只读模式打开该 .pth 文件。
   # torch.load(...)：用 PyTorch 加载保存的模型/训练状态字典，返回的是一个 dict 类型。
        training = Training(device, state['config'] if 'config' in state else config) #这里会调用 Training.__init__() 方法初始化一个新的训练对象。
        training.saver.restore(state)  #这里会调用 Training.__init__() 方法初始化一个新的训练对象。
        print('Configuration')
        training.saver.print_config(offset = 4)       #training.saver.print_config(offset = 4)
        return training

    def initialize(self):
        # Shared network
        self.shared_network = SharedNetwork()
        self.scene_networks = { key:SceneSpecificNetwork(4) for key in TASK_LIST.keys() } #对 TASK_LIST 中的每一个任务 key，创建一个 SceneSpecificNetwork 的实例，并传入参数 4，放入字典中。

        # Share memory
        self.shared_network.share_memory()
        for net in self.scene_networks.values():
            net.share_memory()

        # Callect all parameters from all networks
        parameters = list(self.shared_network.parameters()) # 把共享网络参数放到列表里
        for net in self.scene_networks.values():         
            parameters.extend(net.parameters()) # 把每个场景网络的参数一个一个添加进parameters列表

        # Create optimizer
        optimizer = SharedRMSprop(parameters, eps=self.rmsp_epsilon, alpha=self.rmsp_alpha, lr=self.learning_rate) #用 RMSprop 优化器来训练所有网络
        optimizer.share_memory()

        # Create scheduler
        scheduler = AnnealingLRScheduler(optimizer, self.total_epochs)

        # Create optimizer wrapper     #把优化器和调度器封装到一个 TrainingOptimizer 类中
        optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, scheduler)
        self.optimizer = optimizer_wrapper
        optimizer_wrapper.share_memory()

        # Initialize saver
        self.saver = TrainingSaver(self.shared_network, self.scene_networks, self.optimizer, self.config)
    
    def run(self):
        self.logger.info("Training started") #记录一条日志，说明训练已经开始。这是为了调试和监控用的。

        # Prepare threads
        branches = [(scene, int(target)) for scene in TASK_LIST.keys() for target in TASK_LIST.get(scene)] #生成一个任务列表，每个任务是一个 (scene, target) 的元组
        def _createThread(id, task): #id：这个线程的编号（用于日志标识等）。 #task：一个元组 (scene, target)，表示这个线程要训练的场景和目标任务
            (scene, target) = task #将传入的任务元组 task 拆开
            net = nn.Sequential(self.shared_network, self.scene_networks[scene]) 把共享网络和特定场景的网络串联成一个完整的前向计算模型
            net.share_memory()
            return TrainingThread( #将所有准备好的组件传入 TrainingThread 类，构造一个训练线程对象：
                id = id,
                optimizer = self.optimizer,
                network = net,
                scene = scene,
                saver = self.saver,
                max_t = self.max_t,
                terminal_state_id = target,
                **self.config)

        self.threads = [_createThread(i, task) for i, task in enumerate(branches)] #为每个 (scene, target) 任务创建一个线程，共享同一个优化器和保存器。
        
        try:   #尝试执行 try 代码块中的操作，如果过程中发生了 KeyboardInterrupt（即用户按了 Ctrl+C 中断程序），就会跳转到 except。
            for thread in self.threads: #self.threads 是一个 TrainingThread 实例列表
                thread.start()   #启动每个线程，

            for thread in self.threads:  
                thread.join()   #join() 的作用是：等待该线程执行完毕。 程序会阻塞在这里，直到所有训练线程都运行结束（正常停止）才会继续往下执行。
        except KeyboardInterrupt:
            # we will save the training
            print('Saving training session')
            self.saver.save()
        

    def _init_logger(self):
        logger = logging.getLogger('agent') #logging.getLogger('agent') 会获取一个名为 'agent' 的 logger 实例。## 'agent' 是 logger 的名字，你可以把它理解为给日志打的标签。
        logger.setLevel(logging.INFO) #设置日志级别为 INFO，表示只输出 INFO（信息）WARNING（警告）ERROR（错误）CRITICAL（严重错误）	不会输出 DEBUG（调试）级别的消息。这可以控制日志输出的详细程度。
        logger.addHandler(logging.StreamHandler(sys.stdout)) #添加一个日志处理器：StreamHandler。 StreamHandler(sys.stdout) 表示：将日志信息输出到 终端控制台（标准输出），即你平时运行程序时看到的黑框里。
        return logger
