import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


'''
architect.step()是用来优化alpha的
'''

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  # 参数中的-1就代表这个位置由其他位置的数字来推断
  # 把xs中的每一个元素拉成一行，然后摞成n行
  return torch.cat([x.view(-1) for x in xs])  # 把x先拉成一行，然后把所有的x摞起来，变成n行


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                      lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                      weight_decay=args.arch_weight_decay)  # 用来更新α的optimizer

  """
  我们更新梯度就是theta = theta + v + weight_decay * theta 
    1.theta就是我们要更新的参数
    2.weight_decay*theta为正则化项用来防止过拟合
    3.v的值我们分带momentum和不带momentum：
      普通的梯度下降：v = -dtheta * lr 其中lr是学习率，dx是目标函数对x的一阶导数
      带momentum的梯度下降：v = lr*(-dtheta + v * momentum)
  """
  # 【完全复制外面的Network更新w的过程】，对应公式6第一项的w − ξ*dwLtrain(w, α)
  # 不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开，主要是因为我们这里的更新不能对Network的w进行更新

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    # 训练集输入、标签；验证集输入、标签；学习率、优化器
    self.optimizer.zero_grad()  # 清除上一步的残余更新参数值
    if unrolled:  # 用论文的提出的方法
      self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:  # 不用论文提出的bilevel optimization，只是简单的对α求导
      self._backward_step(input_valid, target_valid)
    self.optimizer.step()  # 应用梯度：根据反向传播得到的梯度进行参数的更新， 这些parameters的梯度是由loss.backward()得到的，optimizer存了这些parameters的指针
    # 因为这个optimizer是针对alpha的优化器，所以他存的都是alpha的参数

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    """
    更新模型参数w为w‘（代码里面的theta就是w）
    """
    loss = self.model._loss(input, target)  # Ltrain(w,α）获取交叉熵损失
    theta = _concat(self.model.parameters()).data  # 把参数整理成一行代表一个参数的形式,得到我们要更新的参数theta
    try:
      # v是每个参数，用每个参数各自的的momentum_buffer都去乘network_momentum（0.9），得到一列，每行都是相应参数对应的moment
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
        self.network_momentum)  # momentum*v,用的就是Network进行w更新的momentum，
    except:
      moment = torch.zeros_like(theta)  # 不加momentum
    dtheta = _concat(torch.autograd.grad(loss,
                                         self.model.parameters())).data + self.network_weight_decay * theta  # 前面的是loss对参数theta求梯度，self.network_weight_decay*theta就是正则项
    # 对参数进行更新，等价于optimizer.step()
    # 就是用theta.sub(eta, moment + dtheta)去替换原来的theta
    # eta是学习率
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(theta[0])
    # s = moment + dtheta
    # print(s[0])
    # print(eta)
    # print(theta.sub(eta, moment + dtheta)[0])
    # # print(theta[0].sub(eta[0],s[0]))
    # # print(theta.size()) torch.Size([1930618])
    # # t = torch.tensor([[1], [2], [3]])
    # # print(t)
    # # t = t.sub(eta, [[0.1], [0.2], [0.3]])
    # # print(t)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # 这里就是theta-eta*(moment + dtheta)
    # 也就是w − ξ*dwLtrain(w, α)
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))  # w − ξ*dwLtrain(w, α)


    return unrolled_model

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()  # 反向传播，计算梯度

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # 计算公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)
    # w' unrolled_model里的w已经是做了一次更新后的w，也就是得到了w'
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    # Lval 对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)
    # 求Lval这个损失关于所有参数的梯度，梯度存在 参数.grad中
    unrolled_loss.backward()
    # dαLval(w',α) 拿出关于结构参数的梯度
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    # dw'Lval(w',α) 拿出关于w’的梯度
    vector = [v.grad.data for v in unrolled_model.parameters()]
    # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    #最后验证集上对alpha的梯度为：.dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    for g, ig in zip(dalpha, implicit_grads):
      # 把dalpha中的，本来是只有dαLval(w',α)，改为dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
      # g和ig是其中的每行的值，也就是每个参数自己的
      # 这整个for循环就相当于dalpha = dalpha - implicit_grads
      g.data.sub_(eta, ig.data)

    # 对α进行更新
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  # 用w − ξ*dwLtrain(w, α)去更新模型里面的w
  def _construct_model_from_theta(self, theta):
    """
    就是用theta-eta*(moment + dtheta)，也就是w − ξ*dwLtrain(w, α)，去替换w，这里新构建了一个模型去往里面的参数赋新值，返回后的模型就是更新w之后的模型了
    """
    model_new = self.model.new()
    model_dict = self.model.state_dict()  # Returns a dictionary containing a whole state of the module.

    params, offset = {}, 0
    for k, v in self.model.named_parameters():  # k是参数的名字，v是参数
      v_length = np.prod(v.size())
      # 将参数k的值更新为theta对应的值
      params[k] = theta[offset: offset + v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)  # 模型中的参数已经更新为做一次反向传播后的值
    model_new.load_state_dict(model_dict)  # 恢复模型中的参数，也就是我新建的mode_new中的参数为model_dict
    return model_new.cuda()

  # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
  def _hessian_vector_product(self, vector, input, target, r=1e-2):  # vector就是dw'Lval(w',α)
    R = r / _concat(vector).norm()  # epsilon

    # dαLtrain(w+,α)
    for p, v in zip(self.model.parameters(), vector):
      # p就是每个参数w'，v就是针对每个w'的dw'Lval(w',α)
      # w = w'+epsilon*dw'Lval(w',α)
      # 即p = p + R * v
      p.data.add_(R, v)  # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)
    # 计算w+、α情况下训练集上的loss关于α的梯度
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    # dαLtrain(w-,α)
    for p, v in zip(self.model.parameters(), vector):
      # 这使得p是p = p + R * v，要得到p - R * v，就要在当前p的基础上减去双倍的R * v
      p.data.sub_(2 * R,
                  v)  # 将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)
    # 计算w-、α情况下训练集上的loss关于α的梯度
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    # 将模型的参数从w-恢复成w
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)  # w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w

    # 这样每一行上都是这个结构参数的(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

