
import datetime
import time
from typing import Any
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import imageio.v2 as imageio
import csv
import json
import re
import folium
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import glob

import transbigdata as tbd


class CommonTimer:
    def __init__(self) -> None:
        self.times = []
        self.start()
        
    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

class SparseMatrix:
    def __init__(self, data, rows, columns) -> None:
        pass
        
    def __call__(self, matrix) -> Any:
        self.Matrix = matrix

    def GetItem(self, row_index, column_index):
        row_start = self.Matrix .indptr[row_index]
        row_end = self.Matrix .indptr[row_index + 1]
        row_values = self.Matrix .data[row_start:row_end]

        index_start = self.Matrix .indptr[row_index]
        index_end = self.Matrix .indptr[row_index + 1]
        row_indices = list(self.Matrix .indices[index_start:index_end])

        value_index = row_indices.index(column_index)

        if value_index >= 0:
            return row_values[value_index]
        else:
            return 0

def DisplayStartInfo(description=""):

    print("-------------------------Start---{}----------------------".format(description))
    startTime = datetime.datetime.now()
    print(startTime.strftime('%Y-%m-%d %H:%M:%S'))
    return startTime


def DisplayCompletedInfo(description="", startTime=datetime.datetime.now(), 
                         isDisplayTimeConsumed=False):

    if isDisplayTimeConsumed==True:
        print('Time consumed:', str(datetime.datetime.now() - startTime).split('.')[0])
    print("Completed at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".\n")
    print("-------------------------Completed---{}----------------------".format(description))

def DisplaySeparator(description=""):
    print("---{}------------------------------------------------------\n".format(description))


def List2Csv(listData, CsvPath, axis=0):

    if axis == 1:
        if len(listData) > 0:
            df = pd.DataFrame(listData)
            df.to_csv(CsvPath, mode='w', encoding='utf-8', header=0, index=0)
    else:
        if len(listData) > 0:
            df = pd.DataFrame([listData])
            df.to_csv(CsvPath, mode='w', encoding='utf-8', header=0, index=0)

def GetMultiplyDatasMaximumValue(Dataset, MaximumValueAmount, axis=0):

    if axis == 0:
        if MaximumValueAmount > Dataset.shape[1]:
            print("CommonCode.py GetMultiplyDatasMaximumValue \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[0] {}.".
                  format(MaximumValueAmount, Dataset.shape[0]))
            return None

        i = 0
        print(Dataset.shape[1])
        MaximunsData = pd.DataFrame(np.zeros((Dataset.shape[0], MaximumValueAmount)), index=Dataset.index)
        for index, row in Dataset.iterrows():
            MaximunsData.iloc[i, :] = pd.DataFrame(row).nlargest(MaximumValueAmount, index, keep='first').T
            i += 1

        return MaximunsData
    elif axis == 1:
        if MaximumValueAmount > Dataset.shape[0]:
            print("CommonCode.py GetMultiplyDatasMaximumValue \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[1] {}.".
                  format(MaximumValueAmount, Dataset.shape[1]))
            return None

        i = 0
        MaximunsData = pd.DataFrame(np.zeros((MaximumValueAmount, Dataset.shape[1])), columns=Dataset.columns)
        for columnName, column in Dataset.items():
            TempColumn = pd.DataFrame(column).nlargest(MaximumValueAmount, columnName, keep='first')

            TempColumn.reset_index(drop=True, inplace=True)
            MaximunsData[MaximunsData.columns[i]] = TempColumn
            i += 1
        return MaximunsData
    else:
        print("Axis is error.")
        return None


def GetMultiplyDataMaximumIndexorColumnName(Dataset, MaximumValueAmount, axis=0):

    if axis == 0:
        if MaximumValueAmount > Dataset.shape[1]:
            print("CommonCode.py GetMultiplyDataMaximumIndexorColumnName \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[0] {}.".
                  format(MaximumValueAmount, Dataset.shape[0]))
            return None

        i = 0
        MaximunsFlag = pd.DataFrame(np.zeros((Dataset.shape[0], MaximumValueAmount)), index=Dataset.index)

        for index, row in Dataset.iterrows():
            MaximunsFlag.iloc[i, :] = pd.DataFrame(row).nlargest(MaximumValueAmount, index, keep='first').T.columns
            i += 1

        return MaximunsFlag
    elif axis == 1:
        if MaximumValueAmount > Dataset.shape[0]:
            print("CommonCode.py GetMultiplyDataMaximumIndexorColumnName \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[1] {}.".
                  format(MaximumValueAmount, Dataset.shape[1]))
            return None

        i = 0
        MaximunsFlag = pd.DataFrame(np.zeros((MaximumValueAmount, Dataset.shape[1])), columns=Dataset.columns)
        for columnName, column in Dataset.items():
            MaximunsFlag[MaximunsFlag.columns[i]] = pd.DataFrame(column).nlargest(MaximumValueAmount, columnName, keep='first').index.index
            print(MaximunsFlag)
            i += 1
        return MaximunsFlag
    else:
        print("Axis is error.")
        return None
    
def GetNonzeroIndexorColumnName(Dataset, axis=0):

    nonZero = []

    DatasetColumnName = Dataset.columns
    DatasetIndexName = Dataset.index

    if axis == 0:
        for index, row in Dataset.iterrows():
            nonZero.append(DatasetColumnName[(np.array(row).nonzero())[0]].tolist())
    elif axis == 1:
        for columnName, column in Dataset.iteritems():
            nonZero.append(DatasetIndexName[(np.array(column).nonzero())[0]].tolist())
    else:
        print('CommonCode.py GetNonzeroIndexorColumnName() Error. input axis {} is Error.'.format(axis))
        return None
    
    nonZero_df = pd.DataFrame(nonZero, index=Dataset.index.tolist())
    return nonZero_df

def ReadHugeFile(inputPath):
    with open(inputPath, 'r', encoding='UTF-8', errors='ignore') as file:
        for line in file:
            try:
                yield line
            except:
                pass


class PrivateDebug():
    
    def __init__(self) -> None:
        self.DisplayLevels = ['All', 'Information', 'Key', 'Debug', 'Warning', 'Error']
        self.Keyword = ""
        
    def AddDisplayLevel(self, Keyword):
        if Keyword not in self.DisplayLevels:
            self.DisplayLevels.append(Keyword)
        return True

    def SetDisplayKeyword(self, keyword):
        self.Keyword = keyword
    
    def OutputContent(self, keyword, msg, *args):

        if keyword.lower() == "nodisplay":
            return 
        if (self.Keyword == keyword):

            print("{} value is {}".format(msg, args))

    
class GenerateAnimation():
    def __init__(self, x, y, z, x_label, y_label, z_label, 
                 title, figureSavePath, figureMainName, gifSavePath, 
                 figsize=(16, 12), dpi=64, duration=0.2, type='3d',
                 startAngle=30, endAngle=70, interval=4) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.title = title
        self.duration = duration
        self.figureSavePath = figureSavePath
        self.figureMainName = figureMainName
        self.gifSavePath = gifSavePath
        self.figsize = figsize
        self.dpi = dpi
        self.type = type
        self.startAngle = startAngle
        self.endAngle = endAngle
        self.interval = interval
        pass
    
    
    def __call__(self):
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False

        ims = []

        for angle in range(self.startAngle, self.endAngle, self.interval):
            plt.clf()

            fig = plt.figure(figsize=(16, 12), dpi=48)
            ax = plt.axes(projection='3d')

            ax.set_title('每年特征统计图')
            ax.set_xlabel('weekofyear')
            ax.set_ylabel('NodeID')
            ax.set_zlabel('statistic')
            ax.axis('auto')

            ax.view_init(30, angle)
            im = ax.scatter3D(self.x, self.y, self.z).findobj()

            pictureName = self.figureSavePath + self.figureMainName + str(angle) + '.png'
            plt.savefig(pictureName, dpi=96)
            ims.append(im)
        
        path = self.figureSavePath
        pictureNames = os.listdir(path)
        list_of_im_paths = []
        for pictureName in pictureNames:
            list_of_im_paths.append(self.figureSavePath + pictureName)

        ims = [imageio.imread(f) for f in list_of_im_paths]
        imageio.mimwrite(self.gifSavePath, ims, duration = self.duration)
        print("Generate Animation has Completed.")
        
def np_3d_to_csv(data, 
                 path, 
                 datatype='float'):
    a2d = data.reshape(data.shape[0], -1)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(a2d)

def np_3d_read_csv(path='./Data/Output/StayMatrx/{}.csv',
                   shape=(-1, 128, 3),
                   datatype='float'):

    with open(path, "r") as f:
        reader = csv.reader(f)
        a2d = np.array(list(reader)).astype(datatype)

    a = a2d.reshape(shape)
    # print(a.shape)
    return a

def data_split_onedimension(sequence, windows_length=100):

    x = []
    y = []
    
    for i in range(len(sequence)):
        labelIndex = i + windows_length
        if labelIndex > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:labelIndex], sequence[labelIndex]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def data_split_twodimension(sequence, windows_length=100, step_length=1):

    x = []
    y = []
    
    for i in range(math.ceil(len(sequence)/step_length)):
        labelIndex = step_length * i + windows_length
        if labelIndex > len(sequence) - 1:
            break
        # sequence[i:labelIndex, :], sequence[labelIndex, :]
        seq_x, seq_y = sequence[step_length*i:labelIndex, :], sequence[labelIndex, :]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def data_split_onedimension(sequence, windows_length=100):

    firstDimension = sequence.shape[0]
    if firstDimension % windows_length != 0:
        sequence = np.pad(sequence, (0, windows_length -(firstDimension % windows_length)),
                            'constant', constant_values=(0, 0))
    x_original = sequence.reshape(-1, windows_length)

    x = np.array(sequence[:-windows_length]).reshape(-1, windows_length)
    y = np.array(sequence[windows_length:]).reshape(-1, windows_length)

    return x_original, x, y


def data_split_twodimension_to_matrix(sequence, windows_length=100):

    firstDimension = sequence.shape[0]
    lastDimension = sequence.shape[-1]

    if firstDimension % windows_length != 0:
        sequence = np.pad(sequence, ((0, windows_length -(firstDimension % windows_length)),(0,0)),
                        'constant', constant_values=(0,0)) 
    
    x_original = sequence.reshape(-1, windows_length, lastDimension)
    x = np.array(sequence[:-windows_length, :]).reshape(-1, windows_length, lastDimension)
    y = np.array(sequence[windows_length:, :]).reshape(-1, windows_length, lastDimension)

    return x_original, x, y

def data_twodimension_to_threedimension_series(sequence, delete_index,windows_length=100, step_length=1):

    x = []
    y = []
    
    for i in range(math.ceil(len(sequence)/step_length)):
        labelIndex = step_length * i + windows_length
        if labelIndex > len(sequence) - 1 :
            break
        seq_x, seq_y = sequence[step_length*i:labelIndex, :], sequence[labelIndex, :]
        seq_x = np.delete(seq_x, obj=delete_index, axis=1)
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def InverseVector(scaler, one_vetcor_of_originalmatrix, vertor):

    temp = np.row_stack((one_vetcor_of_originalmatrix, vertor))
    return scaler.inverse_transform(temp)[-1, :]


def ReadJson(JsonPath='./Parameters.json'):

    with open(JsonPath, 'r', encoding='utf-8') as file:
        json_data = file.read()
        pattern = r'//.*?$|/\*.*?\*/'
        json_data = re.sub(pattern=pattern, repl=' ',
                        string=json_data, flags=re.MULTILINE|re.DOTALL)
        
        parsed_data = json.loads(json_data)
        return parsed_data

class JSONConfig:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_json()

    def _load_json(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def delete(self, key):
        if key in self.data:
            del self.data[key]
            self.save()

def visualizationFunction(history, FirstSavePath, SecondSavePath, preTitle=""):

    keys = []
    for k in history.history.keys():
        keys.append(k)
    values = []
    for v in history.history.values():
        values.append(v)

    epochs = range(len(values[0]))

    plt.plot(epochs, values[0], 'r', label="Training {}".format(keys[0]))
    plt.plot(epochs, values[1], 'b', label="Validation {}".format(keys[1]))
    plt.title(preTitle + " Training and validation {}".format(keys[1]))
    plt.legend()
    plt.savefig(FirstSavePath)
    plt.figure()

    plt.plot(epochs, values[2], 'r', label="Training {}".format(keys[2]))
    plt.plot(epochs, values[3], 'b', label="Validation {}".format(keys[3]))
    plt.title(preTitle + " Training and validation {}".format(keys[3]))
    plt.legend()
    plt.savefig(SecondSavePath)
    plt.figure()
    plt.show()

def DisplaySingleUserHistoryTrajectory(userID, cols=['lat', 'lon'],
                                       dataPath='../Data/Input/Stay/{}.csv',
                                       savePath='../Pictures/Test/{}Traj.html', zoom_start=12):

    data = pd.read_csv(dataPath.format(userID), usecols=cols)
    map = folium.Map(location=[data[cols[0]].mean(), data[cols[1]].mean()], zoom_start=zoom_start, title='{} trajectory'.format(userID))
    trajectory = []
    
    for name, row in data.iterrows():
        trajectory.append([row[cols[0]], row[cols[1]]])

    folium.PolyLine(trajectory, color="blue", weight=2.5, opacity=1).add_to(map)
    map.save(savePath.format(userID))

def ScalerThreeDimensionMatrix(data, feature_range=(-1, 1)):

    dim0, dim1, dim2 = data.shape
    # dim0, dim2

    reshaped_tensor = data.reshape((dim0, dim1 * dim2))
    return MinMaxScaler(feature_range=feature_range).fit_transform(reshaped_tensor).reshape((dim0, dim1, dim2))

def GetTensorBytes(tensor, name='tensor'):

    print('{} memory size is {} bytes.'.format(name, tensor.nelement() * tensor.element_size() /1024/1024 ))


def CantorPairingFunction(x, y):

    if x >= 0:
        x = 2 * x
    else:
        x = 2 * abs(x) - 1
    
    if y >= 0:
        y = 2 * y
    else:
        y = 2 * abs(y) - 1

    return ((x + y) * (x + y + 1) // 2 + y)

def CantorPairingInverseFunction(z):

    if z < 0 :
        print('CantorPairingInverseFunction input z is out of range.')
        return 0, 0
    
    w = (math.sqrt(8 * z + 1) - 1) // 2
    t = w * (w + 1) // 2
    y = z - t
    x = w - y
    
    if x % 2 == 0:
        x = x / 2
    else:
        x = -((x + 1) / 2)
    
    if y % 2 == 0:
        y = y / 2
    else:
        y = -((y + 1) / 2)

    return int(x), int(y)

def GenerateGrid(df, lonColName='loncol', latColName='latcol'):

    df['grid'] = CantorPairingFunction(df[lonColName], df[latColName])
    return df

def RecoverLoncolLatcol(df, gridColName='grid'):

    df['loncol'], df['latcol']= CantorPairingInverseFunction(df[gridColName])
    return df

def Calculate2DConvOutputShape(inputShape, kennel_size, padding, stride, describe=''):

    h_out = math.floor((inputShape[0] - kennel_size[0] + padding[0]) / stride[0]) + 1
    w_out = math.floor((inputShape[1] - kennel_size[1] + padding[1]) / stride[1]) + 1
    print('{} Conv2D Output height is {}, width is {}.'.format(describe, h_out, w_out))
    return (h_out, w_out)

def Calculate2DPoolMaxOutputShape(inputShape, kennel_size, padding, stride, dilation=1, describe=''):

    h_out = math.floor((inputShape[0] + 2 * padding[0] - dilation * (kennel_size[0] - 1) - 1) / stride[0]) + 1
    w_out = math.floor((inputShape[1] + 2 * padding[1] - dilation * (kennel_size[1] - 1) - 1) / stride[1]) + 1
    print('{} PoolMax2D output height is {}, width is {}.'.format(describe, h_out, w_out))
    return (h_out, w_out)

def SwapColumns_TwoDimension(tensor):

    tensor = tensor.clone()
    tensor[[0, 1]] = tensor[[1, 0]]
    return tensor

def SwapColumns_ThreeDimension(tensor):

    tensor = tensor.clone()
    tensor[:, :, [0, 1]] = tensor[:, :, [1, 0]]
    return tensor

def haversine_distance(coords1, coords2):

    R = 6371.0 

    lat1, lon1 = torch.deg2rad(coords1[:, 0]), torch.deg2rad(coords1[:, 1])
    lat2, lon2 = torch.deg2rad(coords2[:, 0]), torch.deg2rad(coords2[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return R * c 

class MultipleInputDataset(Dataset):

    def __init__(self, src, tgt, label):
        super(MultipleInputDataset, self).__init__()
        self.src = torch.tensor(src)
        self.tgt = torch.tensor(tgt)
        self.label = torch.tensor(label)

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):

        return (self.src[index], self.tgt[index], self.label[index])

def GetDataLoader(**kwargs):

    x = torch.load(kwargs['x_path'], weights_only=False)
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)

    y = torch.load(kwargs['y_path'], weights_only=False)
    if not torch.is_tensor(y):
        y = torch.from_numpy(y)
    print("file path {} ,x shape{}, y shape {}.".format(kwargs['x_path'], x.shape, y.shape))

    if kwargs['isMultipleInput'] == True:
        x_2 = torch.load(kwargs['x_2_path'], weights_only=False)
        if not torch.is_tensor(x_2):
            x_2 = torch.from_numpy(x_2)
        dataset = MultipleInputDataset(x, x_2, y)
    else:
        dataset = TensorDataset(x, y)

    train_dataset, test_dataset = random_split(dataset, 
                                           lengths=[int(kwargs['train_size'] * len(dataset)), 
                                                    len(dataset) - int(kwargs['train_size'] * len(dataset))],
                                           generator=torch.Generator().manual_seed(0))
    
    trainLoader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=False, num_workers=0)
    testLodaer = DataLoader(test_dataset, batch_size=kwargs['batch_size'], shuffle=False, num_workers=0)
    return trainLoader, testLodaer


def load_checkpoint(start_epoch, model, optimizer, checkpoint_path):

    checkpoint_pattern = os.path.join(checkpoint_path, 'checkpoint_*.pth')

    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

        print(f"load latest checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"from {start_epoch} start to train...")
    else:
        print('Cant get any checkpoint file. Restart training.')
        start_epoch = 0

    return start_epoch

def DisplayModelTrainResult(logPath, x_col, y_train_col, y_test_col, 
                            plot_label_train, plot_label_test, 
                            x_label, y_label, title, 
                            savePath, extremeFlag='min', offset=1):

    loss_offset_log= pd.read_csv(logPath)

    if extremeFlag == 'min':
        pdextreme = loss_offset_log.iloc[loss_offset_log[y_test_col].idxmin()]
        extremeepoch = pdextreme[x_col]
        extremeTest = pdextreme[y_test_col]
        show_text = f'x:{int(extremeepoch)}\ny:{extremeTest:.2f}'
    else:
        pdextreme = loss_offset_log.iloc[loss_offset_log[y_test_col].idxmax()]
        extremeepoch = pdextreme[x_col]
        extremeTest = pdextreme[y_test_col]
        show_text = f'x:{int(extremeepoch)}\ny:{extremeTest:.2f}'

    plt.figure(figsize=(8, 5))
    plt.plot(loss_offset_log[x_col], loss_offset_log[y_train_col],  linestyle="-", color="b", label=plot_label_train)
    plt.plot(loss_offset_log[x_col], loss_offset_log[y_test_col],  linestyle="-", color="r", label=plot_label_test)

    plt.scatter(extremeepoch, extremeTest, color='red', s=25) 
    if extremeFlag == 'min':
        plt.annotate(show_text, xytext=(extremeepoch-1, extremeTest+extremeTest/10),xy=(extremeepoch, extremeTest))
    else:
        plt.annotate(show_text, xytext=(extremeepoch-1, extremeTest-extremeTest/10),xy=(extremeepoch, extremeTest))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(savePath)
    plt.show()



def GenerateAllGridMapping(Bounds, 
                           mappingColumnName = 'grid_mapping',
                           mappingSavePath='./Data/Output/all_grid_mapping.csv'):

    GeoParameters = tbd.area_to_params(Bounds, accuracy = 1000, method='rect')
    n_lon = int((Bounds[2] - Bounds[0]) / GeoParameters['deltalon'])
    n_lat = int((Bounds[3] - Bounds[1]) / GeoParameters['deltalat'])

    loncols = list(range(n_lon))
    latcols = list(range(n_lat))
    all_grid_df = pd.DataFrame([[lon, lat] for lon in loncols for lat in latcols], columns=['loncol', 'latcol'])

    all_grid_df = all_grid_df.apply(GenerateGrid , lonColName='loncol', latColName='latcol', axis=1)

    GridColumnData = pd.DataFrame(all_grid_df.loc[:, 'grid'])
    GridColumnData.columns = ['grid']
    Grid_duplicated = GridColumnData.drop_duplicates()
    Grid_duplicated = Grid_duplicated.sort_values(by='grid', ascending=True)
    Grid_duplicated = Grid_duplicated.reset_index(drop=True)
    Grid_duplicated[mappingColumnName] = Grid_duplicated.index
    Grid_duplicated[mappingColumnName] += 1
    Grid_duplicated.to_csv(mappingSavePath)
    return Grid_duplicated