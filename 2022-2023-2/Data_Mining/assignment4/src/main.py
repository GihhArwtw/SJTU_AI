"""
Runs MNIST training with differential privacy.

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from privacy_account import get_epsilon

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

# Use to flatten the gradient to a vector
def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param

"""
Using matrix project to compress the gradient matrix
"""

def compress(grad, num_k, power_iter=1):
    u, sigma, v = torch.linalg.svd(grad, full_matrices=False)
    B = v.T[:, :num_k]
    B = B ** power_iter
    G_hat = u[:, :num_k] * sigma[:num_k]

    return B, G_hat

"""
Complete the function of per-example clip
"""

def clip_column(tsr, clip_value=1.0):
    tsr_norm = torch.norm(tsr, dim=1, keepdim=True)
    clipped_norm = torch.clamp(tsr_norm, clip_value)
    clipped_tsr = tsr * (clipped_norm / (tsr_norm + 1e-6))

    return clipped_tsr

def train(args, model, device, train_loader, optimizer, epoch, loss_func, clip_value):
    model.train()
    # criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        batch_grad_list = []
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        if not args.disable_dp:
            with backpack(BatchGrad()):
                loss.backward()
            for p in model.parameters():
                batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))     #compose gradient into Matrix
                del p.grad_batch
            """
            Using project method to compress the gradient
            """
            if args.using_compress:
                
                G_hat = flatten_tensor(batch_grad_list)
                B, G_hat = compress(G_hat, args.project_dims)
                G_hat = clip_column(G_hat, clip_value=args.clip_value)          # clip the gradient
                batch_size = G_hat.shape[0]
                G_hat = G_hat.sum(dim=0, keepdim=True)
                G_hat += torch.rand_like(G_hat) * args.sigma * args.clip_value  # add noise
                G_hat = torch.matmul(G_hat, B.T) / float(batch_size)            # project the gradient

            #per-example clip
            else:
                """
                Complete the code of DPSGD
                """

                G_hat = flatten_tensor(batch_grad_list)
                G_hat = clip_column(G_hat, clip_value=args.clip_value)              # clip the gradient
                batch_size = G_hat.shape[0]
                G_hat = G_hat.sum(dim=0, keepdim=True)
                G_hat += torch.rand_like(G_hat) * args.sigma * args.clip_value      # add noise
                G_hat = G_hat / float(batch_size)

            # update the gradient
            batch_size = G_hat.shape[0]
            start_idx = 0
            for i, p in enumerate(model.parameters()):
                grad = G_hat[:, start_idx:start_idx + p.numel()]                   
                grad = grad.reshape(p.grad.shape)
                start_idx += p.numel()

                p.grad = grad
                
        else:
            loss.backward()
            try:
                for p in model.parameters():
                    del p.grad_batch
            except:
                pass
            
        optimizer.step()
        losses.append(loss.item())
    #get the num of the training dataset from train_loader
    if not args.disable_dp:
        epsilon = get_epsilon(epoch, delta=args.delta, sigma=args.sigma, sensitivity=clip_value, batch_size=args.batch_size, training_nums=len(train_loader)*args.batch_size)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--clip_value",
        type=float,
        default=1.,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--using-compress",
        action="store_true",
        default=False,
        help="Using matrix decomposition to compress the gradient",
    )    
    parser.add_argument(
        "--project-dims",
        type=int,
        default=32,
        help="The gradient project dimension",
    )  
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../dataset/mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20010530,
        help="the random seed",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = SampleConvNet().to(device)
    model = extend(model)
    if not args.disable_dp:
        loss_func = nn.CrossEntropyLoss(reduction='sum')
    else:
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    loss_func = extend(loss_func)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
    
    
    run_results = []
    for _ in range(args.n_runs):
        
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, loss_func, args.clip_value)
            test(model, device, test_loader)
        run_results.append(test(model, device, test_loader))

        print()

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"mnist_{args.lr}_{args.sigma}_"
        f"{args.clip_value}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()