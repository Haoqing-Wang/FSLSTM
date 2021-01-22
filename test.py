import argparse
import torch
from tqdm import tqdm
import numpy as np
import os

from models.ResNet12_embedding import Resnet12
from models.Conv4_embedding import Conv4
from models.classification_heads import ClassificationHead
from utils import set_gpu, count_accuracy, log

def one_hot(indices, depth):
    """
    Inputs:
       indices:  a (n_batch, m) Tensor or (m) Tensor.
       depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()  # (n_batch, m, depth) or (m, depth)
    index = indices.view(indices.size()+torch.Size([1]))  # (n_batch, m, 1) or (m, 1)
    if len(indices.size()) < 2:
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    else:
        encoded_indicies = encoded_indicies.scatter_(2, index, 1)
    return encoded_indicies

def get_model(options): #return (embedding network, classification head)
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
        print("Cannot recognize the network type!")
        assert False

    # Choose the classification head
    cls_head = ClassificationHead(base_learner=options.head).cuda()

    device = (options.gpu).split(',')
    device = [int(s) for s in device]
    network = torch.nn.DataParallel(network, device_ids=device)
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(options, phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(options, phase='test')
        data_loader = FewShotDataloader
    else:
        print("Cannot recognize the dataset type")
        assert(False)
    return (dataset_test, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='Logs/exp_1/best_model.pth', help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=10000, help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5, help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1, help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=15, help='number of query examples per training class')
    parser.add_argument('--psm_iters', type=int, default=0, help='iteration number of PSM')
    parser.add_argument('--network', type=str, default='ResNet12', help='choose which embedding network to use. Conv4, ResNet12')
    parser.add_argument('--head', type=str, default='LSSVM', help='choose which embedding network to use. LSSVM, NN, RR, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet', help='choose which classification head to use. miniImageNet, CIFAR_FS')
    opt = parser.parse_args()
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode)# num of batches per epoch

    set_gpu(opt.gpu)
    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()

    # Evaluate on test set
    test_accuracies = []
    for i, batch in enumerate(tqdm(dloader_test(1)), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query

        emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
        emb_support = emb_support.reshape(1, n_support, -1)
        emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
        emb_query = emb_query.reshape(1, n_query, -1)

        # PSM
        for _ in range(opt.psm_iters):
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)
            labels_pseudo = torch.argmax(logit_query, dim=2)  # (1, n_query)
            labels_pseudo_one_hot = one_hot(labels_pseudo, opt.way)  # (1, n_query, way)
            proto_query = torch.bmm(labels_pseudo_one_hot.transpose(1, 2), emb_query)  # (1, way, d)
            proto_query = proto_query.div(labels_pseudo_one_hot.transpose(1, 2).sum(dim=2, keepdim=True).expand_as(proto_query) + 1e-5)
            labels_pseudo = torch.arange(opt.way).unsqueeze(0).cuda()
            emb_support = torch.cat([emb_support, proto_query], dim=1)
            labels_support = torch.cat([labels_support, labels_pseudo], dim=1)

        logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)
        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())

        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if i % 50 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'.format(i, opt.episode, avg, ci95, acc))