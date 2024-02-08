import torch
from torchvision import datasets, transforms


def load_data(dataset_name, batch_size, seed):
    torch.manual_seed(seed)

    if dataset_name == "MNIST":
        transform = transforms.ToTensor()
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
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