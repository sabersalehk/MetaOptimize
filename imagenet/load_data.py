import torch
from torchvision import datasets, transforms
import os


def load_data(dataset_name, batch_size, seed):
    torch.manual_seed(seed)
    if dataset_name == "MNIST":
        if not os.path.exists("data/MNIST"): os.makedirs("data/MNIST")
        os.symlink(f"/datashare/MNIST", f"data/MNIST/raw")
        transform = transforms.ToTensor()
        trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='CIFAR-10', train=True, download=False, transform=transform)
        testset = datasets.CIFAR10(root='CIFAR-10', train=False, download=False, transform=transform)
    elif dataset_name == "ImageNet":
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        trainset = datasets.ImageFolder(root='ILSVRC2012/train', transform=transform)
        testset = datasets.ImageFolder(root='ILSVRC2012/validation', transform=transform)
        num_workers=4 # can add to the trainloader and testloader as a new attribute
    else: 0/0 # return error
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader



def compute_test_accuracy(net, testloder, device):
    net.eval() # switch to evaluation mode for testing
    correct_predictions_test = 0
    total_predictions_test = 0

    with torch.no_grad():
        for data in testloder:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions_test += labels.size(0)
            correct_predictions_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_predictions_test / total_predictions_test
    net.train() # switch back to training mode
    return test_accuracy