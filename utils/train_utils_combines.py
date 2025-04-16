#!/usr/bin/python
# -*- coding:utf-8 -*-
import logging
import os
import time
import warnings
import torch
from torch import nn
import numpy as np
from torch import optim
from utils.lr_scheduler import *
import models
import datasets
from utils.save import Save_Tool
from loss.DAN import DAN
from loss.DSAN import DSAN
from loss.DAM_CCBL import LDAM_CCBL


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # 路径加载
        if args.data_name == 'CWRU':
            args.data_dir = args.data_dir[0]
        elif args.data_name == 'SQI':
            args.data_dir = args.data_dir[1]
        else:
            None

        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
           print(args.transfer_task)
           args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args, args.normlizetype).data_split(transfer_learning=True)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # Define the model
        self.model = getattr(models, args.model_name)(args.pretrained)

        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        if args.bottleneck:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                              {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]
        else:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'Cos':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0.00005)

        elif args.lr_scheduler == 'transferLearning':
            param_lr = []
            for param_group in self.optimizer.param_groups:
                param_lr.append(param_group["lr"])
            self.lr_scheduler = transferLearning(self.optimizer, param_lr, args.max_epoch)
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model_all.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model_all.load_state_dict(torch.load(args.resume, map_location=args.device))

        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        self.classifier_layer.to(self.device)

        self.adversarial_loss = nn.BCELoss()

        self.criterion = nn.CrossEntropyLoss()

        if args.data_name == 'CWRU':
            self.classes = ['NC', 'IF_7', 'BF_7', 'OF_7']
            self.state_num = 4
        elif args.data_name == 'SQI':
            self.classes = ['NS', 'OF', 'IF', 'BF']
            self.state_num = 4
        else:
            print("Error!")

        # the loss to overcome unbalanced
        if args.data_name == 'CWRU':
            samples_per_cls = np.array(args.source_sample)
        elif args.data_name == 'SQI':
            samples_per_cls = np.array(args.source_sample)
        else:
            print("Error!")

        if args.distance_metric:
            if args.structure_loss == 'MK_MMD':
                self.structure_loss = DAN

            elif args.structure_loss == "DSAN":
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.structure_loss = DSAN

            else:
                raise Exception("loss not implement")

        weights = samples_per_cls.max() / samples_per_cls
        weights = torch.cuda.FloatTensor(weights)

        self.LDAM_CCBL = LDAM_CCBL(num_classes=len(self.classes), class_weights=weights, cls_num_list=samples_per_cls, Lambda=args.Lambda)

    def train(self):

        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        save_list = Save_Tool(max_num=args.max_model_num)
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
                if args.lr_scheduler == 'RL':
                    logging.info('current lr: {}'.format(self.optimizer.param_groups[0]['lr']))
                else:
                    logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])
            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train' or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, target_labels = next(iter_target)
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        features = self.model(inputs)
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)

                        outputs = self.classifier_layer(features)

                        if args.cost_loss:
                            if args.loss == 'LDAM_CCBL':
                                if phase != 'source_train' or epoch < args.middle_epoch:
                                    logits = outputs
                                    loss = self.LDAM_CCBL(logits, labels)
                                else:
                                    logits = outputs.narrow(0, 0, labels.size(0))
                                    classifier_loss = self.LDAM_CCBL(logits, labels)

                        else:
                            if phase != 'source_train' or epoch < args.middle_epoch:
                                logits = outputs
                                loss = self.criterion(logits, labels)

                            else:
                                logits = outputs.narrow(0, 0, labels.size(0))
                                classifier_loss = self.criterion(logits, labels)

                        # -----------------------------------------------------
                        if phase == 'source_train' and epoch >= args.middle_epoch:

                            if args.structure_loss == 'MK_MMD':
                                structure_loss = self.structure_loss(features.narrow(0, 0, labels.size(0)),
                                                                   features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)))

                            elif args.structure_loss == 'DSAN':
                                pseudo_label = self.softmax_layer(outputs)
                                structure_loss = self.structure_loss(
                                    features.narrow(0, 0, args.batch_size),
                                    features.narrow(0, args.batch_size, inputs.size(0) - args.batch_size),
                                    # labels,  target_labels
                                    labels,
                                    pseudo_label.narrow(0, args.batch_size, inputs.size(0) - args.batch_size),
                                    class_num=self.state_num)

                            # Calculate the balance factor
                            if args.trade_off_distance == 'Cons':
                                lam_distance = args.lam_distance
                            elif args.trade_off_distance == 'Step':
                                if args.pretrained == True:
                                    lam_distance = args.micro * (2 / (1 + math.exp(-10 * ((epoch-args.middle_epoch) /
                                                                            (args.max_epoch-args.middle_epoch)))) - 1)
                                else:
                                    lam_distance = args.micro * (2 / (1 + math.exp(-10 * (epoch / args.max_epoch))) - 1)
                            else:
                                raise Exception("balance factor not exist")

                            loss = classifier_loss + lam_distance * structure_loss

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))
                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)

                    # save the best model according to the val accuracy
                    if (epoch_acc >= best_acc or epoch > args.max_epoch-2) and (epoch > args.middle_epoch-1):
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))