# Pytorch / Pytorch-lightning tutorial

## 예제로 배우는 Pytorch

---

### 이미지 분류(Classification)

1. 데이터 셋
    - CIFAR10 데이터 사용
    - 10개의 카테고리로 나누어진 이미지 데이터
    - 3x32x32 크기
    
    ![https://tutorials.pytorch.kr/_images/cifar10.png](https://tutorials.pytorch.kr/_images/cifar10.png)
    
2. 모델 설계

```python
class Net(nn.Module):
    def __init__(self):
        # CNN 2개, FC 3개로 구성
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 2x2 커널을 사용해 max pool
        x = self.pool(F.relu(self.conv2(x))) 
        x = torch.flatten(x, 1) # batch를 제외하고 1차원으로 축소
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 최종적으로 10개의 카테고리에 대한 값 출력
        return x
```

```python
# gpu사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Net().to(device)

# 모델의 전체적인 모습을 보여주는 torchsummary 라이브러리
torchsummary.summary(net, images[0].shape)
```

Out:

```html
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             456
         MaxPool2d-2            [-1, 6, 14, 14]               0
            Conv2d-3           [-1, 16, 10, 10]           2,416
         MaxPool2d-4             [-1, 16, 5, 5]               0
            Linear-5                  [-1, 120]          48,120
            Linear-6                   [-1, 84]          10,164
            Linear-7                   [-1, 10]             850
================================================================
Total params: 62,006
Trainable params: 62,006
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.06
Params size (MB): 0.24
Estimated Total Size (MB): 0.31
----------------------------------------------------------------
```

3. 모델 예측 결과

Out:

```html
Accuracy for class plane is: 56.0 %
Accuracy for class car   is: 55.7 %
Accuracy for class bird  is: 45.7 %
Accuracy for class cat   is: 26.1 %
Accuracy for class deer  is: 52.3 %
Accuracy for class dog   is: 58.1 %
Accuracy for class frog  is: 70.1 %
Accuracy for class horse is: 63.2 %
Accuracy for class ship  is: 70.4 %
Accuracy for class truck is: 73.9 %
```

## Pytorch-lightning으로 코드 간결화

### 이미지 분류

---

1. 모델 설계
- Pytorch

```python
class Net(nn.Module):
    def __init__(self):
        # CNN 2개, FC 3개로 구성
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 2x2 커널을 사용해 max pool
        x = self.pool(F.relu(self.conv2(x))) 
        x = torch.flatten(x, 1) # batch를 제외하고 1차원으로 축소
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 최종적으로 10개의 카테고리에 대한 값 출력
	        return x
```

- Lightning

```python
class Net(pl.LightningModule):
    # 해당 모델을 처음 초기화 할 때 해당 초기화 함수 실행
    def __init__(self):
        # CNN 2개, FC 3개로 구성
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.acc = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1()

    # model(x)의 형태로 모델에 입력 데이터를 넣었을 때 동작
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
		
		# 학습 스텝, batch를 입력으로 받아 loss를 계산 후 리턴
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        return loss

		# 검증 스텝, batch를 입력으로 받아 검증 데이터에 대해 metric value와 loss기록
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc_score = self.acc(output, y)
        f1_score = self.f1(output, y)
        self.log_dict({
            'val_acc': acc_score, 'val_f1': f1_score, 
            'val_loss': loss
        })
		
		# 검증 스텝과 동일. 테스트 데이터에 대해 동작
    def test_step(self, batch, batch_idx):
        x, y = batch 
        output = self(x)
        acc_score = self.acc(output, y)
        f1_score = self.f1(output, y)
        self.log_dict({
            'test_acc': acc_score, 'test_f1': f1_score
        })
    
		# 반드시 오버라이딩 필요. optimizer 설정
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```

2. 학습 코드
- Pytorch

```python
EPOCHS = 2

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # inputs: (batch, channel, width, height)
        # labels: (class)
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000: 0.3f}')
            running_loss = 0.0

print('Finished Training')
```

- Lightning

```python
trainer = pl.Trainer(gpus=1, max_epochs=2)
trainer.fit(net, trainloader, valloader)
```
