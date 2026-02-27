import os
import torch
import torch.nn as nn
import torch.optim as optim


import torchvision.datasets as datasets
import torchvision.transforms as transforms


from PIL import Image
import numpy as np
import time
import zipfile
import matplotlib.pyplot as plt



def parse_label_from_filename(filename):
    parts = filename.split('_')
    gender = '여성' if parts[-1].startswith('W') else '남성'  
    style = parts[-2]  
    label = f"{gender}_{style}"  # 성별과 스타일을 결합하여 라벨 생성
    return label

def load_data_and_labels(data_dir):
    data = []
    labels = []
    label_map = {}  # 각 라벨에 대해 고유 인덱스 매핑
    current_index = 0

    for root, _, files in os.walk(data_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            label = parse_label_from_filename(file_name)

            # 라벨인덱스 부여
            if label not in label_map:
                label_map[label] = current_index
                current_index += 1

            image = Image.open(file_path).resize((224, 224))
            image = np.array(image) / 255.0 
            data.append(image)
            labels.append(label_map[label]) 

    data = np.array(data)
    labels = np.array(labels)
    return data, labels, label_map

# 사용 예시
train_data_dir = '/content/drive/MyDrive/sorted_training'
test_data_dir = '/content/drive/MyDrive/sorted_validation'

train_data, train_labels, train_label_map = load_data_and_labels(train_data_dir)
test_data, test_labels, test_label_map = load_data_and_labels(test_data_dir)

print("학습 데이터 크기:", train_data.shape)
print("학습 라벨:", train_labels)
print("라벨 맵핑:", train_label_map)



# 하이퍼파라미터
batch_size = 32

# 데이터
path = 'data' 
train_dir = '/content/drive/MyDrive/sorted_training'
valid_dir = '/content/drive/MyDrive/sorted_validation'

# 이미지 전처리
train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

valid_transforms = transforms.Compose([
    transforms.ToTensor(),
])


# ImageFolder
train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)

# 클래스별 샘플 수
class_sample_counts = [len([s for s in train_data.samples if s[1] == i]) for i in range(len(train_data.classes))]
weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
sample_weights = [weights[label] for _, label in train_data.samples]

# WeightedRandomSampler 정의
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 데이터 로더
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=True)

# 데이터셋 크기
print(f'Train images: {len(train_loader.dataset)}')
print(f'Validation images: {len(valid_loader.dataset)}')



# 시드 고정 제거
random_nums = [np.random.randint(0, len(train_loader.dataset)) for i in range(5)]  
plt.figure(figsize=(10, 2))

for i, idx in enumerate(random_nums):
    image, label = train_loader.dataset[idx]  
    image = np.transpose(image.numpy(), (1, 2, 0))  

    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.title(f'Label: {train_loader.dataset.classes[label]}')  
    plt.axis('off')

plt.show()



import torch.nn.init as init

# 가중치 초기화 함수
def initialize_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        init.xavier_uniform_(m.weight) 
        if m.bias is not None:
            init.zeros_(m.bias)  # 편향
    elif isinstance(m, torch.nn.BatchNorm2d):
        init.ones_(m.weight)  
        init.zeros_(m.bias)   


import torch

# train_loader에서 클래스 개수 동적으로 설정
num_classes = len(train_loader.dataset.classes) 

class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # 2개의 합성곱 블록 정의
        self.conv_block1, self.shape = self.conv_block(224, 3, 8, 3, padding=1)
        self.conv_block2, self.shape = self.conv_block(self.shape, 8, 16, 3, stride=2)

        # 3개의 완전 연결 블록 정의
        self.fc_block1 = self.fc_block(16 * self.shape**2, 256) 
        self.fc_block2 = self.fc_block(256, 128)
        self.fc_block3 = self.fc_block(128, 32)

        # 출력층을 클래스 수에 맞게 설정
        self.output = torch.nn.Linear(32, num_classes)

        # 가중치 초기화 적용
        self.apply(self.initialize_weights)

    # 가중치 초기화 함수
    def initialize_weights(self, m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight) 
            torch.nn.init.zeros_(m.bias)  

    # 합성곱 블록 정의 함수
    def conv_block(self, shape, in_, out_, kernel, stride=1, padding=0):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_, out_, kernel, stride=stride, padding=padding, bias=False),
            torch.nn.BatchNorm2d(out_),
            torch.nn.ReLU()
        )
        shape = int(np.floor((shape - kernel + 2 * padding) / stride) + 1) 
        return block, shape

    # 완전 연결 블록 정의 함수
    def fc_block(self, in_, out_):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_, out_, bias=False),
            torch.nn.BatchNorm1d(out_),
            torch.nn.ReLU()
        )
        return block

    # 순전파 함수
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = torch.flatten(x, 1) 
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)

        x = self.output(x)
        return x  # 최종 출력값 반환

model = CNN(num_classes=num_classes)


import torch.optim as optim

# device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 클래스 수를 동적으로 설정하여 모델 초기화
num_classes = len(train_loader.dataset.classes) 
model = CNN(num_classes=num_classes).to(device)

# 옵티마이저 설정
loss_func = nn.CrossEntropyLoss()  

# 학습률 스케줄러
learning_rate = 0.002  # 학습률 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습률 조정
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=False) 
model



# 모델 초기화 및 가중치 초기화 적용
num_classes = len(train_loader.dataset.classes)
model = CNN(num_classes=num_classes).to(device)

# 가중치 초기화 함수 수정
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # Kaiming 
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Xavier
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm
        nn.init.ones_(m.weight)  
        nn.init.zeros_(m.bias)   

model.apply(initialize_weights) 


def train(model, epochs):
    # 손실함수와 옵티마이저
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    losses = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        print('Epoch: {:0>4}...'.format(epoch+1), end='\t')

        # 모델을 학습 모드로 설정
        model.train()
        running_loss = 0.0
        running_acc = 0

        # 학습 데이터 로더에서 배치 단위로 학습
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 변화도 초기화
            optimizer.zero_grad()

            # 예측
            preds = model(images)

            # 손실 계산
            loss = criterion(preds, labels)

            # 정확도 계산
            _, predicted_labels = torch.max(preds, 1) 
            running_acc += (predicted_labels == labels).sum().item() 

            # 역전파
            loss.backward()
            # 가중치 갱신
            optimizer.step()

            # 손실값 갱신
            running_loss += loss.item()

        # 학습 손실 및 정확도 계산
        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader.dataset)

        losses['train_loss'].append(train_loss)
        losses['train_acc'].append(train_acc)

        print('train_loss: {:.4f}'.format(train_loss), end='\t')
        print('train_acc: {:.4f}'.format(train_acc), end='\t\t')

        # 검증 성능 평가
        val_loss, val_acc = evaluate(model, valid_loader, criterion)

        losses['val_loss'].append(val_loss)
        losses['val_acc'].append(val_acc)

        print('val_loss: {:.4f}'.format(val_loss), end='\t')
        print('val_acc: {:.4f}'.format(val_acc))
        print()

    return losses



##학습

# SimplifiedCNN
num_classes = len(train_loader.dataset.classes) 
model = CNN(num_classes=num_classes).to(device) 

# 모델 학습
losses = train(model, 1)


##정확도
plt.figure(figsize=(14, 6))
plt.set_cmap('Paired')

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(losses['train_loss'], label='Train Loss')
plt.plot(losses['val_loss'], label='Validation Loss')
plt.legend()
plt.grid(True)
plt.title('Loss Graph')

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(losses['train_acc'], label='Train Accuracy')
plt.plot(losses['val_acc'], label='Validation Accuracy')
plt.legend()
plt.grid(True)
plt.title('Accuracy Graph')



plt.show()
