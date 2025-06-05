import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import os
from matplotlib.font_manager import FontProperties

def image_resize(images):
    # preprocess the shape of data to the size of (36, 36)
    TARGET_SIZE = (36, 36)
    images_ = []
    for img in images:
        image = cv2.resize(img / img.max(), dsize=(TARGET_SIZE[0], TARGET_SIZE[1]), interpolation=cv2.INTER_CUBIC)
        images_.append(image)
    return np.asarray(images_).astype("float32")

df=pd.read_pickle(os.path.join(os.getcwd(), "data/MIR-WM811K/Python/WM811K.pkl"))

#list the field name of the structure
df.info()

# get the failure wafer map
failure_type = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
df_failure = df[df['failureType'].isin(failure_type)].loc[:, ['waferMap', 'failureType', 'trainTestLabel']]

# transform the failure_type into digital label
group_labels = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
                'Near-full': 7}
df_failure['Label'] = df_failure['failureType'].map(group_labels)
# data = df_failure.loc[:, ['waferMap', 'Label', 'trainTestLabel']]

# Select training and test data
trainData_temp=df_failure[df_failure['trainTestLabel']=='Training'].reset_index()    # train set：17625
testData_temp=df_failure[df_failure['trainTestLabel']=='Test'].reset_index()         # test set：7894

# obtain the dataset for federated training
trainData = trainData_temp.loc[:, ['waferMap', 'Label']]
testData = testData_temp.loc[:, ['waferMap', 'Label']]

X_train = trainData.iloc[:, 0].values
# X_train = image_resize(trainData.iloc[:, 0].values)
y_train = trainData.iloc[:, 1].values
X_test = testData.iloc[:, 0].values
# X_test = image_resize(testData.iloc[:, 0].values)
y_test = testData.iloc[:, 1].values

data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

with open("wm811k-unprocessed.pkl", "wb") as file:
    pickle.dump(data, file)
print("Save successfully！")

# print("X_train: mean({}), std({})".format(X_train.mean(), X_train.std()))
# print("X_test: mean({}), std({})".format(X_test.mean(), X_test.std()))
#
# uniqueType=trainData_temp['failureType'].unique()
# uniqueType.sort()
# print(uniqueType)
#
# #Plot a wafer map for each type
# font = FontProperties(family='Times New Roman', size=16)
# fig, axes = plt.subplots(2, 4, figsize=(10, 5))
# for i in range(8):
#     idx = trainData[trainData['Label'] == i].index
#     exampleIdx = idx[0]
#
#     ax = axes[i // 4, i % 4]    # get the subgragh
#     image = trainData.iloc[exampleIdx]['waferMap']
#     ax.imshow(image)
#     ax.set_title(failure_type[i], fontproperties=font)
#     ax.axis('off')

# plt.subplots_adjust(wspace=0.1, hspace=0.2)
# plt.savefig("wafer_map.pdf", bbox_inches='tight')
# plt.show()
#
# df_none = df[df['failureType'] == 'none']
# idx = df[df['failureType'] == 'none'].index
# plt.imshow(df_none.iloc[idx[0]]['waferMap'])
# plt.show()


