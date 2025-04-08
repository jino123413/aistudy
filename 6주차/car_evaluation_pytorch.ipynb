{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "bafa41bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader, TensorDataset\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from torchsummary import summary\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "95c0d737",
   "metadata": {},
   "outputs": [],
   "source": [
    "url = \"https://raw.githubusercontent.com/MyungKyuYi/AI-class/main/diabetes.csv\"\n",
    "data = pd.read_csv(url)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "dc5cc004",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6</td>\n",
       "      <td>148</td>\n",
       "      <td>72</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>33.6</td>\n",
       "      <td>0.627</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>763</th>\n",
       "      <td>10</td>\n",
       "      <td>101</td>\n",
       "      <td>76</td>\n",
       "      <td>48</td>\n",
       "      <td>180</td>\n",
       "      <td>32.9</td>\n",
       "      <td>0.171</td>\n",
       "      <td>63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>764</th>\n",
       "      <td>2</td>\n",
       "      <td>122</td>\n",
       "      <td>70</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>36.8</td>\n",
       "      <td>0.340</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>765</th>\n",
       "      <td>5</td>\n",
       "      <td>121</td>\n",
       "      <td>72</td>\n",
       "      <td>23</td>\n",
       "      <td>112</td>\n",
       "      <td>26.2</td>\n",
       "      <td>0.245</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>766</th>\n",
       "      <td>1</td>\n",
       "      <td>126</td>\n",
       "      <td>60</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.1</td>\n",
       "      <td>0.349</td>\n",
       "      <td>47</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>767</th>\n",
       "      <td>1</td>\n",
       "      <td>93</td>\n",
       "      <td>70</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>30.4</td>\n",
       "      <td>0.315</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>768 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "0              6      148             72             35        0  33.6   \n",
       "1              1       85             66             29        0  26.6   \n",
       "2              8      183             64              0        0  23.3   \n",
       "3              1       89             66             23       94  28.1   \n",
       "4              0      137             40             35      168  43.1   \n",
       "..           ...      ...            ...            ...      ...   ...   \n",
       "763           10      101             76             48      180  32.9   \n",
       "764            2      122             70             27        0  36.8   \n",
       "765            5      121             72             23      112  26.2   \n",
       "766            1      126             60              0        0  30.1   \n",
       "767            1       93             70             31        0  30.4   \n",
       "\n",
       "     DiabetesPedigreeFunction  Age  Outcome  \n",
       "0                       0.627   50        1  \n",
       "1                       0.351   31        0  \n",
       "2                       0.672   32        1  \n",
       "3                       0.167   21        0  \n",
       "4                       2.288   33        1  \n",
       "..                        ...  ...      ...  \n",
       "763                     0.171   63        0  \n",
       "764                     0.340   27        0  \n",
       "765                     0.245   30        0  \n",
       "766                     0.349   47        1  \n",
       "767                     0.315   23        0  \n",
       "\n",
       "[768 rows x 9 columns]"
      ]
     },
     "execution_count": 109,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "7ce64a6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='count'>"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig_size = plt.rcParams[\"figure.figsize\"]\n",
    "fig_size[0] = 8\n",
    "fig_size[1] = 6\n",
    "plt.rcParams[\"figure.figsize\"] = fig_size# 2개의 고유 값이 있는 경우\n",
    "data.Outcome.value_counts().plot(kind='pie', autopct='%0.05f%%', \n",
    "                                colors=['lightblue', 'lightgreen'], \n",
    "                                explode=(0.05, 0.05))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "701780b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',\n",
       "       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "6c48cf5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',\n",
    "       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "04e615a1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "58b0a84e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6</td>\n",
       "      <td>148</td>\n",
       "      <td>72</td>\n",
       "      <td>35</td>\n",
       "      <td>0</td>\n",
       "      <td>33.6</td>\n",
       "      <td>0.627</td>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>183</td>\n",
       "      <td>64</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>0.672</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>168</td>\n",
       "      <td>43.1</td>\n",
       "      <td>2.288</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>763</th>\n",
       "      <td>10</td>\n",
       "      <td>101</td>\n",
       "      <td>76</td>\n",
       "      <td>48</td>\n",
       "      <td>180</td>\n",
       "      <td>32.9</td>\n",
       "      <td>0.171</td>\n",
       "      <td>63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>764</th>\n",
       "      <td>2</td>\n",
       "      <td>122</td>\n",
       "      <td>70</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>36.8</td>\n",
       "      <td>0.340</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>765</th>\n",
       "      <td>5</td>\n",
       "      <td>121</td>\n",
       "      <td>72</td>\n",
       "      <td>23</td>\n",
       "      <td>112</td>\n",
       "      <td>26.2</td>\n",
       "      <td>0.245</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>766</th>\n",
       "      <td>1</td>\n",
       "      <td>126</td>\n",
       "      <td>60</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.1</td>\n",
       "      <td>0.349</td>\n",
       "      <td>47</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>767</th>\n",
       "      <td>1</td>\n",
       "      <td>93</td>\n",
       "      <td>70</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "      <td>30.4</td>\n",
       "      <td>0.315</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>768 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "0              6      148             72             35        0  33.6   \n",
       "1              1       85             66             29        0  26.6   \n",
       "2              8      183             64              0        0  23.3   \n",
       "3              1       89             66             23       94  28.1   \n",
       "4              0      137             40             35      168  43.1   \n",
       "..           ...      ...            ...            ...      ...   ...   \n",
       "763           10      101             76             48      180  32.9   \n",
       "764            2      122             70             27        0  36.8   \n",
       "765            5      121             72             23      112  26.2   \n",
       "766            1      126             60              0        0  30.1   \n",
       "767            1       93             70             31        0  30.4   \n",
       "\n",
       "     DiabetesPedigreeFunction  Age  Outcome  \n",
       "0                       0.627   50        1  \n",
       "1                       0.351   31        0  \n",
       "2                       0.672   32        1  \n",
       "3                       0.167   21        0  \n",
       "4                       2.288   33        1  \n",
       "..                        ...  ...      ...  \n",
       "763                     0.171   63        0  \n",
       "764                     0.340   27        0  \n",
       "765                     0.245   30        0  \n",
       "766                     0.349   47        1  \n",
       "767                     0.315   23        0  \n",
       "\n",
       "[768 rows x 9 columns]"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "84a4a54f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 데이터와 타겟 분리\n",
    "X = data.drop('Outcome', axis=1).values\n",
    "y = data['Outcome'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "644b8da5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standardize the data\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(X)\n",
    "\n",
    "# Split the dataset into training and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "9d25c686",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((614, 8), (154, 8), (614,), (154,))"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, X_test.shape, y_train.shape, y_test.shape, "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "c1bf4be5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert to PyTorch tensors\n",
    "X_train = torch.tensor(X_train, dtype=torch.float32)\n",
    "y_train = torch.tensor(y_train, dtype=torch.int64)\n",
    "X_test = torch.tensor(X_test, dtype=torch.float32)\n",
    "y_test = torch.tensor(y_test, dtype=torch.int64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "7c5743a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Create DataLoader\n",
    "train_dataset = TensorDataset(X_train, y_train)\n",
    "train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
    "\n",
    "test_dataset = TensorDataset(X_test, y_test)\n",
    "test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "b32ba6de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(torch.Size([614, 8]),\n",
       " torch.Size([154, 8]),\n",
       " torch.Size([614]),\n",
       " torch.Size([154]))"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, X_test.shape, y_train.shape, y_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29cb1afb",
   "metadata": {},
   "source": [
    "# 모델 정의"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "edf1d642",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CarEvaluationDense(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(CarEvaluationDense, self).__init__()\n",
    "        self.fc1 = nn.Linear(8, 64)  # Change from 6 to 8 to match your input data\n",
    "        self.fc2 = nn.Linear(64, 32)\n",
    "        self.fc3 = nn.Linear(32, 4)  # 4 classes in the dataset\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x = torch.relu(self.fc1(x))\n",
    "        x = torch.relu(self.fc2(x))\n",
    "        x = self.fc3(x)\n",
    "        return x\n",
    "\n",
    "# Initialize the model, loss function, and optimizer\n",
    "model = CarEvaluationDense()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9be587e7",
   "metadata": {},
   "source": [
    "# 손실 함수 및 최적화 기법 정의"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "32ea43f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.001)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3d0cbeb0",
   "metadata": {},
   "source": [
    "# 모델 학습"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "1b022524",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20, Loss: 1.2539, Accuracy: 70.13%\n",
      "Epoch 2/20, Loss: 0.9288, Accuracy: 75.32%\n",
      "Epoch 3/20, Loss: 0.6861, Accuracy: 78.57%\n",
      "Epoch 4/20, Loss: 0.5692, Accuracy: 81.17%\n",
      "Epoch 5/20, Loss: 0.5426, Accuracy: 81.17%\n",
      "Epoch 6/20, Loss: 0.5095, Accuracy: 83.12%\n",
      "Epoch 7/20, Loss: 0.5005, Accuracy: 81.82%\n",
      "Epoch 8/20, Loss: 0.4842, Accuracy: 81.17%\n",
      "Epoch 9/20, Loss: 0.4840, Accuracy: 82.47%\n",
      "Epoch 10/20, Loss: 0.4744, Accuracy: 81.17%\n",
      "Epoch 11/20, Loss: 0.4849, Accuracy: 81.17%\n",
      "Epoch 12/20, Loss: 0.4585, Accuracy: 81.82%\n",
      "Epoch 13/20, Loss: 0.4565, Accuracy: 81.82%\n",
      "Epoch 14/20, Loss: 0.4662, Accuracy: 79.87%\n",
      "Epoch 15/20, Loss: 0.4516, Accuracy: 81.17%\n",
      "Epoch 16/20, Loss: 0.4415, Accuracy: 80.52%\n",
      "Epoch 17/20, Loss: 0.4558, Accuracy: 80.52%\n",
      "Epoch 18/20, Loss: 0.4420, Accuracy: 81.17%\n",
      "Epoch 19/20, Loss: 0.4350, Accuracy: 81.17%\n",
      "Epoch 20/20, Loss: 0.4386, Accuracy: 80.52%\n",
      "Training complete.\n"
     ]
    }
   ],
   "source": [
    "# Variables to store loss and accuracy\n",
    "train_losses = []\n",
    "test_accuracies = []\n",
    "\n",
    "# Training loop\n",
    "num_epochs = 20\n",
    "for epoch in range(num_epochs):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for inputs, labels in train_dataloader:\n",
    "        # Zero the parameter gradients\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        # Forward pass\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "\n",
    "        # Backward pass and optimize\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        running_loss += loss.item()\n",
    "\n",
    "    # Calculate average loss over an epoch\n",
    "    train_losses.append(running_loss / len(train_dataloader))\n",
    "\n",
    "    # Evaluate on test data\n",
    "    model.eval()\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in test_dataloader:\n",
    "            outputs = model(inputs)\n",
    "            _, predicted = torch.max(outputs.data, 1)\n",
    "            total += labels.size(0)\n",
    "            correct += (predicted == labels).sum().item()\n",
    "\n",
    "    accuracy = 100 * correct / total\n",
    "    test_accuracies.append(accuracy)\n",
    "\n",
    "    print(f\"Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy:.2f}%\")\n",
    "\n",
    "print(\"Training complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ee551f8",
   "metadata": {},
   "source": [
    "# 모델 평가"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "8160ffb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[94 13]\n",
      " [17 30]]\n",
      "F1 Score: 0.80\n",
      "Precision: 0.80\n",
      "Recall: 0.81\n",
      "Specificity: 0.76\n"
     ]
    }
   ],
   "source": [
    "# Evaluation\n",
    "model.eval()\n",
    "all_labels = []\n",
    "all_predictions = []\n",
    "with torch.no_grad():\n",
    "    for inputs, labels in test_dataloader:\n",
    "        outputs = model(inputs)\n",
    "        _, predicted = torch.max(outputs.data, 1)\n",
    "        all_labels.extend(labels.cpu().numpy())\n",
    "        all_predictions.extend(predicted.cpu().numpy())\n",
    "\n",
    "# Convert to numpy arrays\n",
    "all_labels = np.array(all_labels)\n",
    "all_predictions = np.array(all_predictions)\n",
    "\n",
    "# Calculate metrics\n",
    "conf_matrix = confusion_matrix(all_labels, all_predictions)\n",
    "f1 = f1_score(all_labels, all_predictions, average='weighted')\n",
    "precision = precision_score(all_labels, all_predictions, average='weighted')\n",
    "recall = recall_score(all_labels, all_predictions, average='weighted')\n",
    "\n",
    "# Calculate specificity for each class\n",
    "specificity = []\n",
    "for i in range(conf_matrix.shape[0]):\n",
    "    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])\n",
    "    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]\n",
    "    specificity.append(tn / (tn + fp))\n",
    "\n",
    "print(f'Confusion Matrix:\\n{conf_matrix}')\n",
    "print(f'F1 Score: {f1:.2f}')\n",
    "print(f'Precision: {precision:.2f}')\n",
    "print(f'Recall: {recall:.2f}')\n",
    "print(f'Specificity: {np.mean(specificity):.2f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "263e0aec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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mR3QSa/zRbwtcR+AhNAAJIQFIDrIDz4d95wXQqDRbLEl2peTbjNqG1thaDTjUse/V5b56a3mACApMpBJupdraGlDVV0LACDehX/EIiKi/otJOhFRP6LxUWJsXDDGXjFNXo6C/NWYnBSByUn22xvNbSiqaexI2utxriN5P1fdAGNTK8qMzSgzNmP3mYt2x6mUEvRBfmhutaC2odXuWvi+UCokBPmpEBygRrC/CgmhA663ieQhijqmugf7q6Dzd/8ftIiIyPsxSSciIo/hr/ZBil6LFH3XBVdqG8w4V1PfZfS9sKYBLW3WLgvrqX0UCPZXIdhf3f4IUCHIX40QfzWCOraHBFx+HhygRqDGx3a7Obo+gwcPRnFxcZftjz/+ONasWYPm5mY8/fTT2LhxI1paWjB9+nS8++67iIhwz4yFopr2zw2nuhMRkaswSSciIq8QHKDGuIAQjBsUYrfdahUoNzXj/KVGBGh8bEm3v1rJe167wcGDB2GxXL504Pjx45g6dSruueceAMDvfvc7bNu2DZs2bYJOp8OyZctw1113Yffu3W6p723DQvHZb9PRz26GQ0REbsQknYiIvJpCISE6yK9f3dJNzsLCwuxev/LKK0hMTMTEiRNhNBqxbt06fPzxx5g8eTIAIDMzE8nJydi3bx9uvvlml9dX66vC+PiQqxckIiJyEIW7K0BERET9k9lsxkcffYRFixZBkiQcOnQIra2tyMjIsJVJSkpCXFwc9u7d2+u5WlpaYDKZ7B5ERESeiEk6ERERucWWLVtgMBiwcOFCAEBFRQXUajWCgoLsykVERKCioqLXc61evRo6nc72iI2NdVKtiYiInItJOhEREbnFunXrMHPmTOj1+p99rmeffRZGo9H2KC0tdUANiYiIXI/XpBMREZHLFRcXIysrC59//rltW2RkJMxmMwwGg91oemVlJSIjI3s9n0ajgUajcVZ1iYiIXIYj6URERORymZmZCA8Px6xZs2zbxo0bB5VKhezsbNu2goIClJSUID093R3VJCIicjmOpBMREZFLWa1WZGZmYsGCBfDxufyriE6nw+LFi7F8+XKEhIRAq9XiiSeeQHp6ultWdiciInIHJulERETkUllZWSgpKcGiRYu67HvjjTegUChw9913o6WlBdOnT8e7777rhloSERG5hySEEO6uhCuZTCbodDoYjUZotVp3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",
      "text/plain": [
       "<Figure size 1200x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot the loss and accuracy\n",
    "plt.figure(figsize=(12, 5))\n",
    "\n",
    "# Plot loss\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(train_losses, label='Training Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.title('Training Loss Over Epochs')\n",
    "plt.legend()\n",
    "\n",
    "# Plot accuracy\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(test_accuracies, label='Test Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy (%)')\n",
    "plt.title('Test Accuracy Over Epochs')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "838a65d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 데이터와 타겟 분리\n",
    "X = data.drop('Outcome', axis=1).values\n",
    "y = data['Outcome'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "c9b536d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standardize the data\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "115e7879",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_array = np.hstack((X, y.reshape(-1, 1)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "a2421c2f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(768, 9)"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_array.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "8ac76e97",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split sequences function\n",
    "def split_sequences(sequences, n_steps):\n",
    "    X, y = list(), list()\n",
    "    for i in range(len(sequences)):\n",
    "        end_ix = i + n_steps\n",
    "        if end_ix > len(sequences):\n",
    "            break\n",
    "        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]\n",
    "        X.append(seq_x)\n",
    "        y.append(seq_y)\n",
    "    return np.array(X), np.array(y)\n",
    "\n",
    "# Apply sequence transformation\n",
    "n_steps = 5\n",
    "X, y = split_sequences(data_array, n_steps)\n",
    "\n",
    "# Split the dataset into training and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "f4b4b83f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((611, 5, 8), (611,), (153, 5, 8), (153,))"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, y_train.shape, X_test.shape, y_test.shape, "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "624607fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert to PyTorch tensors\n",
    "X_train = torch.tensor(X_train, dtype=torch.float32)\n",
    "y_train = torch.tensor(y_train, dtype=torch.int64)\n",
    "X_test = torch.tensor(X_test, dtype=torch.float32)\n",
    "y_test = torch.tensor(y_test, dtype=torch.int64)\n",
    "\n",
    "# Create DataLoader\n",
    "train_dataset = TensorDataset(X_train, y_train)\n",
    "train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
    "\n",
    "test_dataset = TensorDataset(X_test, y_test)\n",
    "test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "39c53795",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CarEvaluationCNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(CarEvaluationCNN, self).__init__()\n",
    "        # Change input channels from 6 to 8\n",
    "        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)\n",
    "        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)\n",
    "        # The rest of your model definition...\n",
    "        # Calculate the correct size for the first linear layer based on your architecture\n",
    "        self.fc1 = nn.Linear(16 * 2, 64)  # Adjust this size based on your pooling and input size\n",
    "        self.fc2 = nn.Linear(64, 32)\n",
    "        self.fc3 = nn.Linear(32, 4)  # 4 classes in the dataset\n",
    "    \n",
    "    def forward(self, x):\n",
    "        # Conv layers\n",
    "        x = torch.relu(self.conv1(x))\n",
    "        x = self.pool(x)\n",
    "        \n",
    "        # Flatten\n",
    "        x = x.view(x.size(0), -1)\n",
    "        \n",
    "        # Fully connected layers\n",
    "        x = torch.relu(self.fc1(x))\n",
    "        x = torch.relu(self.fc2(x))\n",
    "        x = self.fc3(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "9e17eec7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the model, loss function, and optimizer\n",
    "model = CarEvaluationCNN()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "id": "618cf76b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "----------------------------------------------------------------\n",
      "        Layer (type)               Output Shape         Param #\n",
      "================================================================\n",
      "            Conv1d-1                [-1, 16, 5]             400\n",
      "         MaxPool1d-2                [-1, 16, 2]               0\n",
      "            Linear-3                   [-1, 64]           2,112\n",
      "            Linear-4                   [-1, 32]           2,080\n",
      "            Linear-5                    [-1, 4]             132\n",
      "================================================================\n",
      "Total params: 4,724\n",
      "Trainable params: 4,724\n",
      "Non-trainable params: 0\n",
      "----------------------------------------------------------------\n",
      "Input size (MB): 0.00\n",
      "Forward/backward pass size (MB): 0.00\n",
      "Params size (MB): 0.02\n",
      "Estimated Total Size (MB): 0.02\n",
      "----------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "# Print the summary of the model\n",
    "summary(model, input_size=(8, 5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "id": "b2c1d4af",
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
    "\n",
    "# Variables to store loss and accuracy\n",
    "train_losses = []\n",
    "test_accuracies = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "28174255",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20, Loss: 1.3035, Accuracy: 47.06%\n",
      "Epoch 2/20, Loss: 0.9696, Accuracy: 62.75%\n",
      "Epoch 3/20, Loss: 0.6969, Accuracy: 62.75%\n",
      "Epoch 4/20, Loss: 0.6468, Accuracy: 62.09%\n",
      "Epoch 5/20, Loss: 0.6159, Accuracy: 62.75%\n",
      "Epoch 6/20, Loss: 0.6302, Accuracy: 60.78%\n",
      "Epoch 7/20, Loss: 0.6122, Accuracy: 64.05%\n",
      "Epoch 8/20, Loss: 0.5738, Accuracy: 64.05%\n",
      "Epoch 9/20, Loss: 0.5709, Accuracy: 63.40%\n",
      "Epoch 10/20, Loss: 0.5520, Accuracy: 60.13%\n",
      "Epoch 11/20, Loss: 0.5469, Accuracy: 60.13%\n",
      "Epoch 12/20, Loss: 0.5451, Accuracy: 64.05%\n",
      "Epoch 13/20, Loss: 0.5105, Accuracy: 63.40%\n",
      "Epoch 14/20, Loss: 0.4921, Accuracy: 61.44%\n",
      "Epoch 15/20, Loss: 0.4626, Accuracy: 60.78%\n",
      "Epoch 16/20, Loss: 0.4784, Accuracy: 62.75%\n",
      "Epoch 17/20, Loss: 0.4679, Accuracy: 64.05%\n",
      "Epoch 18/20, Loss: 0.4535, Accuracy: 63.40%\n",
      "Epoch 19/20, Loss: 0.4667, Accuracy: 65.36%\n",
      "Epoch 20/20, Loss: 0.4275, Accuracy: 64.05%\n",
      "Training complete.\n",
      "Confusion Matrix:\n",
      "[[78 18]\n",
      " [37 20]]\n",
      "F1 Score: 0.62\n",
      "Precision: 0.62\n",
      "Recall: 0.64\n",
      "Specificity: 0.58\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1200x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Training loop\n",
    "num_epochs = 20\n",
    "for epoch in range(num_epochs):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    for inputs, labels in train_dataloader:\n",
    "        # Zero the parameter gradients\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        # Forward pass\n",
    "        inputs = inputs.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "\n",
    "        # Backward pass and optimize\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        running_loss += loss.item()\n",
    "\n",
    "    # Calculate average loss over an epoch\n",
    "    train_losses.append(running_loss / len(train_dataloader))\n",
    "\n",
    "    # Evaluate on test data\n",
    "    model.eval()\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    all_labels = []\n",
    "    all_predictions = []\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in test_dataloader:\n",
    "            inputs = inputs.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)\n",
    "            outputs = model(inputs)\n",
    "            _, predicted = torch.max(outputs.data, 1)\n",
    "            total += labels.size(0)\n",
    "            correct += (predicted == labels).sum().item()\n",
    "            all_labels.extend(labels.cpu().numpy())\n",
    "            all_predictions.extend(predicted.cpu().numpy())\n",
    "\n",
    "    accuracy = 100 * correct / total\n",
    "    test_accuracies.append(accuracy)\n",
    "\n",
    "    print(f\"Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy:.2f}%\")\n",
    "\n",
    "print(\"Training complete.\")\n",
    "\n",
    "# Convert lists to numpy arrays\n",
    "all_labels = np.array(all_labels)\n",
    "all_predictions = np.array(all_predictions)\n",
    "\n",
    "# Calculate metrics\n",
    "conf_matrix = confusion_matrix(all_labels, all_predictions)\n",
    "f1 = f1_score(all_labels, all_predictions, average='weighted')\n",
    "precision = precision_score(all_labels, all_predictions, average='weighted')\n",
    "recall = recall_score(all_labels, all_predictions, average='weighted')\n",
    "\n",
    "# Calculate specificity for each class\n",
    "specificity = []\n",
    "for i in range(conf_matrix.shape[0]):\n",
    "    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])\n",
    "    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]\n",
    "    specificity.append(tn / (tn + fp))\n",
    "\n",
    "# Print metrics\n",
    "print(f'Confusion Matrix:\\n{conf_matrix}')\n",
    "print(f'F1 Score: {f1:.2f}')\n",
    "print(f'Precision: {precision:.2f}')\n",
    "print(f'Recall: {recall:.2f}')\n",
    "print(f'Specificity: {np.mean(specificity):.2f}')\n",
    "\n",
    "# Plot the loss and accuracy\n",
    "plt.figure(figsize=(12, 5))\n",
    "\n",
    "# Plot loss\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(train_losses, label='Training Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.title('Training Loss Over Epochs')\n",
    "plt.legend()\n",
    "\n",
    "# Plot accuracy\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(test_accuracies, label='Test Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy (%)')\n",
    "plt.title('Test Accuracy Over Epochs')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
