import torch
import time
from training_tools import AverageMeter, accuracy_topk

class Attack(torch.nn.Module):
  def __init__(self, attack_init):
    super(Attack, self).__init__()
    self.attack = torch.nn.Parameter(attack_init, requires_grad=True)

  def forward(self, X, trained_model):
    X_attacked = X + self.attack
    trained_model.eval()
    y = trained_model(X_attacked)
    return y

def clip_params(model, epsilon):
    old_params = {}

    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    old_params['attack'][old_params['attack']>epsilon] = epsilon
    old_params['attack'][old_params['attack']<(-1*epsilon)] = -1*epsilon

    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])

def train_pgd(train_loader, model, criterion, optimizer, epoch, print_freq, epsilon, trained_model, device='cuda'):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var, trained_model)
        loss = criterion(output, target_var)
        loss_neg = -1*loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_neg.backward()
        optimizer.step()

        # clip the parameters
        clip_params(model, epsilon)

        output = output.float()
        loss = loss_neg.float()
        # measure accuracy and record loss
        prec1 = accuracy_topk(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))

def eval_attack(test_loader, model, criterion, trained_model, device):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var, trained_model)
            loss = criterion(output, target_var)
            loss_neg = -1 * loss

            output = output.float()
            loss = loss_neg.float()

            # measure accuracy and record loss
            acc1 = accuracy_topk(output.data, target, k=1)
            acc5 = accuracy_topk(output.data, target, k=5)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

    print('Test\t  Prec@1: {top1.avg:.3f} (Err: {error:.3f} )\n'
          .format(top1=top1,error=100-top1.avg))
    print('Test\t  Prec@5: {top5.avg:.3f} (Err: {error:.3f} )\n'
        .format(top5=top5,error=100-top5.avg))

    return (top1.avg, top5.avg)

def train_pgd_indv(train_loader, attack_models, criterion, optimizers, epoch, print_freq, epsilon, trained_model, device):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (input, target) in enumerate(train_loader):

        attack_model = attack_models[i]
        attack_model.train()

        optimizer = optimizers[i]

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = attack_model(input_var, trained_model)
        loss = criterion(output, target_var)
        loss_neg = -1*loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_neg.backward()
        optimizer.step()

        # clip the parameters
        clip_params(attack_model, epsilon)

        output = output.float()
        loss = loss_neg.float()
        # measure accuracy and record loss
        prec1 = accuracy_topk(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))

def eval_attack_indv(test_loader, attack_models, criterion, trained_model, device):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):

            attack_model = attack_models[i]
            attack_model.eval()

            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = attack_model(input_var, trained_model)
            loss = criterion(output, target_var)
            loss_neg = -1 * loss

            output = output.float()
            loss = loss_neg.float()

            # measure accuracy and record loss
            acc1 = accuracy_topk(output.data, target, k=1)
            acc5 = accuracy_topk(output.data, target, k=5)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

    print('Test\t  Prec@1: {top1.avg:.3f} (Err: {error:.3f} )\n'
          .format(top1=top1,error=100-top1.avg))
    print('Test\t  Prec@5: {top5.avg:.3f} (Err: {error:.3f} )\n'
        .format(top5=top5,error=100-top5.avg))

    return (top1.avg, top5.avg)
