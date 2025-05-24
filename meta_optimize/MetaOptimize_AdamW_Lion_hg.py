import torch
import torch.nn as nn

class MetaOptimize_AdamW_Lion_hg():
    def __init__(self, params, 
                 meta_stepsize = 1e-3, 
                 alpha0 = 1e-3,             # conservative initialization for Adam
                 stepsize_blocks ='scalar', # options: 'scalar', 'layerwise', or a list of integers indicating 
                                            # number of params in each block (e.g., [2,4,5] means 3 bocks containing 2, 3, and 5 param-groups respectively)
                 gamma = 0.9999,
                 b1=0.9, b2=0.999,          # base-adamw parameters
                 meta_b1=0.99, meta_c1=0.9, # meta-lion parameters
                 weight_decay=0.0):
        
        self.params = list(params)
        self.meta_stepsize = meta_stepsize
        self.gamma = gamma
        self.alpha0 = alpha0
        self.b1 = b1
        self.b2 = b2
        self.meta_b1 = meta_b1
        self.meta_c1 = meta_c1
        self.eps = 1e-8
        self.weight_decay = weight_decay

        self.state = {}
        for p in self.params:
            self.state[p] = {
                'step': 0,
                'base_momentum': torch.zeros_like(p.data),
                'base_normalizer': torch.zeros_like(p.data),
                'h': torch.zeros_like(p.data),
                
            }

        num_layers_in_each_block = ([len(self.params)] if  stepsize_blocks=='scalar' else
                                    [1 for _ in self.params] if  stepsize_blocks=='layerwise' else
                                    stepsize_blocks
                                    )
        

        self.meta_state = [{
            'num_layers_in_this_block':block_size,
            'meta_momentum': torch.tensor(0.0, dtype=torch.float32).to(p.data.device),
            'beta':torch.log(torch.tensor(alpha0, dtype=torch.float32)).to(p.data.device),
            } for block_size in num_layers_in_each_block]
        
        if not sum(x['num_layers_in_this_block'] for x in self.meta_state)==len(self.params):
            AssertionError('blocksizes do not match the layers')

    @torch.no_grad()
    def step(self):
        remaining_layers_in_this_block = 0
        block = -1
        for p in self.params:
            if p.grad is None:
                continue
            
            if remaining_layers_in_this_block == 0: # initiate meta paramteres of next block
                block+=1
                meta_state = self.meta_state[block]
                meta_grad = 0.0
                beta = meta_state['beta']
                alpha = torch.exp(beta)
                meta_momentum = meta_state['meta_momentum']
                remaining_layers_in_this_block = meta_state['num_layers_in_this_block'] + 0

            grad = p.grad.data
            state = self.state[p]

            base_momentum = state['base_momentum']
            base_normalizer = state['base_normalizer']
            h = state['h']
            state['step']+=1
            _b1_ = (1-self.b1)/(1-self.b1**state['step'])
            _b2_ = (1-self.b2)/(1-self.b2**state['step'])

            #------------------
            ### Base Update:
            base_momentum.mul_(1-_b1_).add_(grad, alpha=_b1_)    # equivalent to m_hat
            base_normalizer.mul_(1-_b2_).addcmul_(grad, grad, value=_b2_)    # equivalent to v_hat

            if self.weight_decay > 0:
                p.data.add_(p.data, alpha=-alpha*self.weight_decay)
            p.data.add_(base_momentum/(base_normalizer.sqrt()+self.eps), alpha=-alpha)

            # Meta-gradient
            meta_grad += (h * grad).sum()
            h.mul_(self.gamma*(1-alpha*self.weight_decay)).add_(grad/(base_normalizer.sqrt()+self.eps), alpha=alpha)

            state['base_momentum'] = base_momentum
            state['base_normalizer'] = base_normalizer
            state['h'] = h
            
            remaining_layers_in_this_block -=1

            if remaining_layers_in_this_block == 0: # meta update
                meta_momentum = self.meta_b1 * meta_momentum + (1 - self.meta_b1) * meta_grad
                meta_update = self.meta_c1 * meta_momentum + (1 - self.meta_c1) * meta_grad
                meta_sign = meta_update.sign()
                beta.add_(self.meta_stepsize * meta_sign)

                meta_state['meta_momentum'] = meta_momentum
                meta_state['beta'] = beta


    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()




# -----------------------------------------------------------------------------

# --- Simple MLP model for MNIST ---
def get_mlp(num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

# --- Load MNIST data ---
def get_mnist_dataloaders():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader

# --- Evaluation function ---
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100. * correct / total



if __name__ == "__main__":
    # Metaparameters
    epochs = 50
    meta_stepsize = 1e-3
    alpha0 = 1e-3
    stepsize_blocks = 'scalar' # 'scalar', 'blockwise', or [2, 2, 2]
    b1, b2 = 0.9, 0.999
    meta_b1, meta_c1 = 0.99, 0.9
    weight_decay = 0.1
    gamma = 0.9999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data and model
    trainloader, testloader = get_mnist_dataloaders()
    model = get_mlp(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = MetaOptimize_AdamW_Lion_hg(params=model.parameters(),
                                           meta_stepsize=meta_stepsize,
                                           alpha0=alpha0,
                                           stepsize_blocks=stepsize_blocks,
                                           gamma=gamma,
                                           b1=b1, b2=b2,
                                           meta_b1=meta_b1,
                                           meta_c1=meta_c1,
                                           weight_decay=weight_decay)
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100. * correct / total
        test_loss, test_acc = evaluate(model, testloader, device, criterion)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
    
