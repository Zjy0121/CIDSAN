import os
from scipy.io import loadmat
import pandas as pd
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

dataname= {0: ["97.mat", "105.mat", "118.mat", "130.mat"],  # 1797rpm
           1: ["98.mat", "106.mat", "119.mat", "131.mat"],  # 1772rpm
           2: ["99.mat", "107.mat", "120.mat", "132.mat"],  # 1750rpm
           3: ["100.mat", "108.mat", "121.mat", "133.mat"]}  # 1730rpm

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
axis = ["_DE_time", "_FE_time", "_BA_time"]

labels = [i for i in range(0, 4)]


def source_get_files(root, N, args):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data_train = []
    lab_train = []
    data_test = []
    lab_test = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
               path1 = os.path.join(root, datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root, datasetname[0], dataname[N[k]][n])
            data1_train, lab1_train, data1_test, lab1_test = source_data_load(path1, dataname[N[k]][n], label=labels[n], args=args)
            data_train += data1_train
            lab_train += list(lab1_train)
            data_test += data1_test
            lab_test += list(lab1_test)

    return [data_train, lab_train], [data_test, lab_test]


def target_get_files(root, N, args):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data_train = []
    lab_train = []
    data_test = []
    lab_test = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
               path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1_train, lab1_train, data1_test, lab1_test = target_data_load(path1,dataname[N[k]][n],label=labels[n], args=args)
            data_train += data1_train
            lab_train += list(lab1_train)
            data_test += data1_test
            lab_test += list(lab1_test)

    return [data_train, lab_train], [data_test, lab_test]


def source_data_load(filename, axisname, label, args):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]

    data = []
    lab = []
    j = 0
    start, end = 0, args.signal_size
    while end <= fl.shape[0]:
        start = j * args.sample_stride
        end = args.signal_size + j * args.sample_stride
        data.append(fl[start:end])
        lab.append(label)
        start += args.signal_size
        end += args.signal_size
        j += 1

    random.seed(args.seed)
    random.shuffle(data)

    data_train = data[:args.source_sample[label]]
    data_test = data[args.source_sample[label]:args.source_sample[label] + args.test_sample]
    label_train = lab[:args.source_sample[label]]
    label_test = lab[args.source_sample[label]:args.source_sample[label] + args.test_sample]

    return data_train, label_train, data_test, label_test


def target_data_load(filename, axisname, label, args):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]

    data = []
    lab = []
    j = 0
    start, end = 0, args.signal_size
    while end <= fl.shape[0]:
        start = j * args.sample_stride
        end = args.signal_size + j * args.sample_stride
        data.append(fl[start:end])
        lab.append(label)
        start += args.signal_size
        end += args.signal_size
        j += 1

    random.seed(args.seed)
    random.shuffle(data)

    data_train = data[:args.target_sample[label]]
    data_test = data[args.target_sample[label]:args.target_sample[label] + args.test_sample]
    label_train = lab[:args.target_sample[label]]
    label_test = lab[args.target_sample[label]:args.target_sample[label] + args.test_sample]
    return data_train, label_train, data_test, label_test


# --------------------------------------------------------------------------------------------------------------------
class CWRU(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, args, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.args = args
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            train_data, val_data = source_get_files(self.data_dir, self.source_N, self.args)
            train_data_pd = pd.DataFrame({"data": train_data[0], "label": train_data[1]})
            val_data_pd = pd.DataFrame({"data": val_data[0], "label": val_data[1]})
            source_train = dataset(list_data=train_data_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_data_pd, transform=self.data_transforms['val'])

            # get target train and val
            train_data, val_data = target_get_files(self.data_dir, self.target_N, self.args)
            train_data_pd = pd.DataFrame({"data": train_data[0], "label": train_data[1]})
            val_data_pd = pd.DataFrame({"data": val_data[0], "label": val_data[1]})
            target_train = dataset(list_data=train_data_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            return None