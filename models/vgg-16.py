import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class ApproximateConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ApproximateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # mean of absolute values
        mu_w = torch.mean(torch.abs(self.weights))
        
        #  perform min multiplication for approximation
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        w_unf = self.weights.view(self.weights.size(0), -1)
        
        # Apply to input and weights
        min_val, _ = torch.min(x_unf.unsqueeze(1), w_unf.unsqueeze(2))
        min_val = min_val.view(x.size(0), self.out_channels, -1)
        
        # Approximate conv output
        z_approx = mu_w * torch.sum(min_val, dim=2)
        z_approx = z_approx.view(x.size(0), self.out_channels, int((x.size(2) + 2 * self.padding - self.kernel_size) / self.stride + 1), int((x.size(3) + 2 * self.padding - self.kernel_size) / self.stride + 1))
        
        return z_approx

class VGG16Approx(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16Approx, self).__init__()
        self.features = nn.Sequential(
            ApproximateConv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ApproximateConv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ApproximateConv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ApproximateConv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ApproximateConv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ApproximateConv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Training for CIFAR-10
def train_vgg16_approx():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = VGG16Approx().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(10):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

           
            running_loss += loss.item()
            if i % 100 == 99:    
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

if __name__ == "__main__":
    train_vgg16_approx()
