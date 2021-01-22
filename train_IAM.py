import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from models.classification_heads import ClassificationHead
from models.IAM import InverseAttentionModule
from models.ResNet12_embedding import Resnet12
from models.Conv4_embedding import Conv4
from utils import set_gpu, Timer, count_accuracy, check_dir, log

def one_hot(indices, depth):
    """
    Inputs:
       indices:  a (n_batch, m) Tensor or (m) Tensor.
       depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda() #(n_batch, m, depth) or (m, depth)
    index = indices.view(indices.size()+torch.Size([1])) #(n_batch, m, 1) or (m, 1)
    if len(indices.size()) < 2:
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    else:
        encoded_indicies = encoded_indicies.scatter_(2, index, 1)
    return encoded_indicies

def get_model(options): # return (embedding network, classification head)
    # Choose the embedding network
    if options.network == 'Conv4':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = Conv4(avg_pool=True).cuda()
        else:
            network = Conv4().cuda()
    elif options.network == 'ResNet12':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = Resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
        else:
            network = Resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        print ("Cannot recognize the network type!")
        assert (False)

    # Choose the classification head
    cls_head = ClassificationHead(base_learner=opt.head).cuda()

    IAM = InverseAttentionModule(options.dim, reduction=options.reduction, dropout=0.5).cuda()
    device = (options.gpu).split(',')
    device = [int(s) for s in device]
    network = torch.nn.DataParallel(network, device_ids=device)
    IAM = torch.nn.DataParallel(IAM, device_ids=device)
    return (network, cls_head, IAM)

def get_dataset(options): #return (dataset_train, dataset_val, data_loader)
    # Choose the dataset
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(options, phase='train')
        dataset_val = MiniImageNet(options, phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(options, phase='train')
        dataset_val = CIFAR_FS(options, phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type!")
        assert(False)
    return (dataset_train, dataset_val, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=60, help='number of training epochs')
    parser.add_argument('--save_epoch', type=int, default=10,help='frequency of model saving')
    parser.add_argument('--train_shot', type=int, default=5, help='number of support examples per training class')
    parser.add_argument('--val_shot', type=int, default=5, help='number of support examples per validation class')
    parser.add_argument('--train_query', type=int, default=6, help='number of query examples per training class')
    parser.add_argument('--val_episode', type=int, default=2000, help='number of episodes per validation')
    parser.add_argument('--val_query', type=int, default=15, help='number of query examples per validation class')
    parser.add_argument('--train_way', type=int, default=5, help='number of classes in one training episode')
    parser.add_argument('--val_way', type=int, default=5, help='number of classes in one validation episode')
    parser.add_argument('--dim', type=int, default=2048, help='embedding dim')
    parser.add_argument('--reduction', type=int, default=8, help='reduction')
    parser.add_argument('--save_path', default='Logs')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--IAM_lr', type=float, default=0.01, help='learning rate of IAM')
    parser.add_argument('--lr', type=float, default=0.005, help='lr')
    parser.add_argument('--pretrain', type=str, default='Logs/pretrain/Conv4_32_best.pth', help='pretrain dir')
    parser.add_argument('--network', type=str, default='ResNet12', help='choose which embedding network to use. ResNet12, Conv4')
    parser.add_argument('--head', type=str, default='LSSVM', help='choose which classification head to use. LSSVM, NN, RR, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet', help='choose which classification head to use. miniImageNet, CIFAR_FS')
    parser.add_argument('--episodes_per_batch', type=int, default=8, help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.1, help='epsilon of label smoothing')
    opt = parser.parse_args()

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000)  # num of batches per epoch

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.val_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.val_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode)  # num of batches per epoch

    set_gpu(opt.gpu)
    check_dir('Logs/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head, IAM) = get_model(opt)
    pretrained_dict = torch.load(opt.pretrain)['state_dict']
    model_dict = embedding_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    embedding_net.load_state_dict(model_dict)

    optimizer = torch.optim.SGD([{'params':embedding_net.parameters(),'lr':opt.lr},{'params':cls_head.parameters()},{"params":IAM.parameters(),'lr':opt.IAM_lr}], lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, opt.num_epoch+1):
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(epoch, epoch_learning_rate))

        _, _, _ = [x.train() for x in (embedding_net, cls_head, IAM)]
        train_accuracies = []
        train_losses = []
        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))  # (batch_size*train_n_support, d)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))  # (batch_size*train_n_query, d)
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)  # (batch_size, train_n_support, d)
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)  # (batch_size, train_n_query, d)
            emb_support = IAM(emb_support, emb_query, emb_query, labels_support)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)  # (batch_size, train_n_query, train_way)
            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)  # (batch_size*train_n_query, train_way)
            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1) #(batch_size*train_n_query, train_way)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} %'.format(epoch, i, len(dloader_train), loss.item(), train_acc_avg))
        # Train on the training split
        lr_scheduler.step()
        # Evaluate on the validation split
        _, _, _ = [x.eval() for x in (embedding_net, cls_head, IAM)]
        val_accuracies = []
        val_losses = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
                data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
                test_n_support = opt.val_way * opt.val_shot
                test_n_query = opt.val_way * opt.val_query

                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)
                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)
                emb_support = IAM(emb_support, emb_query, emb_query, labels_support)

                logit_query = cls_head(emb_query, emb_support, labels_support, opt.val_way, opt.val_shot)
                loss = x_entropy(logit_query.reshape(-1, opt.val_way), labels_query.reshape(-1))
                acc = count_accuracy(logit_query.reshape(-1, opt.val_way), labels_query.reshape(-1))

                val_accuracies.append(acc.item())
                val_losses.append(loss.item())
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)
        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(), 'IAM': IAM.state_dict()}, os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'.format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'.format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(), 'IAM': IAM.state_dict()}, os.path.join(opt.save_path, 'last_epoch.pth'))
        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict(), 'IAM': IAM.state_dict()}, os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))
        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))