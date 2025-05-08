import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print('Initiating project ☺')

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Prepare data
class CustomImageSet(Dataset):
    def __init__(self, image_directory, excel_path, tensor_ready=True):
        self.tensor_ready = tensor_ready
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((78,808)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        df = pd.read_excel(excel_path, header=None).dropna(subset=[0])
        df[0] = df[0].astype(int)
        label_dict = {row[0]: str(row[1]) for _, row in df.iterrows()}

        items = []
        for fn in os.listdir(image_directory):
            name, ext = os.path.splitext(fn)
            if ext.lower() in ('.png','.jpg','.jpeg'):
                try:
                    file_id = int(name)
                    if file_id in label_dict:
                        path = os.path.join(image_directory, fn)
                        items.append((file_id, path, label_dict[file_id]))
                except ValueError:
                    continue

        items.sort(key=lambda x: x[0])

        # 4) Unzip into aligned lists
        self.ids, self.image_paths, self.labels = zip(*items)
        # Convert back to lists
        self.image_paths = list(self.image_paths)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if self.tensor_ready:
            img = self.transform(img)
            label = torch.tensor([char_to_idx[c] for c in label], dtype=torch.long)

        return img, label

vocabulary = ['<BLANK>']
vocabulary = vocabulary + [
            'a','b','c','d','e','f','g','h','i',
            'j','k','l','m','n','o','p','q','r',
            's','t','u','v','w','x','y','z',
            'A','B','C','D','E','F','G','H','I',
            'J','K','L','M','N','O','P','Q','R',
            'S','T','U','V','W','X','Y','Z','0',
            '1','2','3','4','5','6','7','8','9',
            ',','.',' ','+','-','(',')','/',"'",
            '!','?','='
            ]
vocab_size = len(vocabulary)
print(f'vocab_size:{vocab_size}')
ctc_blank_idx = len(vocabulary)  # reserve the last index for CTC blank
idx_to_char = {i: c for i, c in enumerate(vocabulary)}
char_to_idx = {c: i for i, c in enumerate(vocabulary)}
def str_indices(str):
    return torch.tensor([char_to_idx[c] for c in str], dtype=torch.long)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.maxPool = nn.MaxPool2d(
            kernel_size=2
        )
        self.cnn1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.cnn2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.cnn3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
    def forward(self, x):
        x = self.cnn1(x)
        x = self.maxPool(x)
        x = self.cnn2(x)
        x = self.maxPool(x)
        x = self.cnn3(x)
        #print(f'CNN OUT SHAPE:{x.shape}')
        return x

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size=1024,num_layers=4):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=608,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout = 0.20,
            bidirectional=True
            )
    def forward(self,x):
        out, (h_n, _) = self.lstm(x)
        return out

class LinearDecoder(nn.Module):
    def __init__(self,in_features,vocab_size):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(
            in_features=in_features,
            out_features=vocab_size
            )
    def forward(self,x):
        out = self.decoder(x)
        return out

class totalModel(nn.Module):
    def __init__(self):
        super(totalModel,self).__init__()
        self.CNN_Layer = CNN()
        self.RNN_Layer = RNN(input_size=32) #Final output size of CNN layer
        #2 * Hidden Features, LSTM is bidirectional, +1 for CTC
        self.Decoder_Layer = LinearDecoder(in_features=2*1024,vocab_size=vocab_size+1)
    def forward(self,x):
        x = self.CNN_Layer(x)  # (B, C, H, W)
        x = x.permute(0, 3, 1, 2)         # (B, W, C, H)
        B, W, C, H = x.shape
        x = x.contiguous().view(B, W, C * H)  # (B, W, Features)
        x = self.RNN_Layer(x)
        x = self.Decoder_Layer(x)
        return x
    
loss_function = nn.CTCLoss(blank=0,reduction='mean',zero_infinity=True)

showTestImage = False
if showTestImage:
    debug_image_set = CustomImageSet(
        excel_path='Project Data Labels.xlsx',
        image_directory='Project Divided Data',
        tensor_ready=False)
    image,title=debug_image_set[len(debug_image_set)-1]
    cv2.imshow(title,image)
    cv2.waitKey(0) &0xFF == ord('q') 

def ctc_collate_fn(batch):
    images, labels = zip(*batch)
    image_batch = torch.stack(images)  # (B, C, H, W)
    
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)  # Flattened targets

    return image_batch, labels_concat, label_lengths

def decode_string(x, blank=0):
    prediction_indices = torch.argmax(x, dim=-1)  # (B, T)
    predicted_text = []

    for seq in prediction_indices:
        decoded = []
        prev = None
        for idx in seq:
            idx = idx.item()
            if idx != prev:
                if idx != blank and idx < len(idx_to_char):
                    decoded.append(idx_to_char[idx])
            prev = idx
        predicted_text.append("".join(decoded))

    return predicted_text

tensorData = CustomImageSet(
        excel_path='Project Data Labels.xlsx',
        image_directory='Project Divided Data'
        )
train_set_size = int(len(tensorData)*0.8)
valid_set_size = int(len(tensorData)*0.15)
test_set_size = len(tensorData)-train_set_size-valid_set_size

train_set, valid_set, test_set = data.random_split(tensorData,[train_set_size,valid_set_size,test_set_size],\
                                              generator=torch.Generator().manual_seed(42))
# # Manually split
# train_set = torch.utils.data.Subset(tensorData, range(0, train_set_size))
# valid_set = torch.utils.data.Subset(tensorData, range(train_set_size, train_set_size + valid_set_size))
# test_set = torch.utils.data.Subset(tensorData, range(train_set_size + valid_set_size, len(tensorData)))
# # Manually split
# train_set = torch.utils.data.Subset(tensorData, range(len(tensorData)-train_set_size, len(tensorData)))
# valid_set = torch.utils.data.Subset(tensorData, range(len(tensorData)-train_set_size-valid_set_size, len(tensorData)-train_set_size))
# test_set = torch.utils.data.Subset(tensorData, range(0, test_set_size))

#batch_size = train_set_size
batch_size = 3
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=ctc_collate_fn)
# train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn)

#Begin training
model = totalModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.000065)
epochs = 110

train_loss = []
ave_loss = []
for epoch in range(epochs):
    model.train()
    print(f'Epoch is:{epoch}')
    for x,y,y_lens in train_dataloader:
        #Move to GPU
        x,y,y_lens = x.to(device),y.to(device),y_lens.to(device)
        
        model.zero_grad()
        y_hat = model(x)
        string_out = decode_string(y_hat)
        print(string_out)
        y_hat = y_hat.permute(1, 0, 2)  # (T, N, C) for CTC
        probability = nn.functional.log_softmax(y_hat,dim=-1)
        input_lengths = torch.full((x.size(0),), probability.size(0), dtype=torch.long).to(device)
        #print(f'Probability shape:{probability.shape},y shape:{y.shape}')
        loss = loss_function(probability,y,input_lengths,y_lens)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    ave_loss.append(np.mean(train_loss))
    print(f'Average loss over epoch:{ave_loss[-1]}')
    train_loss = []

    #Validation
    print('Beginning validation')
    model.eval()

    with torch.no_grad():
        valid_loss = []
        for x,y,y_lens in valid_dataloader:
            #Move to GPU
            x,y,y_lens = x.to(device),y.to(device),y_lens.to(device)
            
            y_hat = model(x)
            string_out = decode_string(y_hat)
            print(string_out)
            y_hat = y_hat.permute(1, 0, 2)  # (T, N, C) for CTC
            probability = nn.functional.log_softmax(y_hat,dim=-1)
            input_lengths = torch.full((x.size(0),), probability.size(0), dtype=torch.long).to(device)
            loss = loss_function(probability,y,input_lengths,y_lens)
            valid_loss.append(loss.item())
        print(f'Average validation loss:{np.mean(valid_loss)}')

model.eval()
print('Beginning Test')
with torch.no_grad():
    test_loss = []
    for x,y,y_lens in test_dataloader:
        #Move to GPU
        x,y,y_lens = x.to(device),y.to(device),y_lens.to(device)
        
        y_hat = model(x)
        string_out = decode_string(y_hat)
        print(string_out)
        y_hat = y_hat.permute(1, 0, 2)  # (T, N, C) for CTC
        probability = nn.functional.log_softmax(y_hat,dim=-1)
        input_lengths = torch.full((x.size(0),), probability.size(0), dtype=torch.long).to(device)
        loss = loss_function(probability,y,input_lengths,y_lens)
        test_loss.append(loss.item())
    print(f'Average test loss:{np.mean(test_loss)}')