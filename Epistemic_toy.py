import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import tensorflow as tf
from torchvision.utils import save_image
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
learning_rate = 1e-3
test_trials = 20

parser = argparse.ArgumentParser(description='practice') 
parser.add_argument("--type", type=str, default='train')
parser.add_argument("--model", type=str, default='model.pth')
parser.add_argument("--batchsize", type=int, default=1)


args = parser.parse_args()

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

testdataset = torchvision.datasets.MNIST(root='data',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

testdata_loader = torch.utils.data.DataLoader(dataset=testdataset,
                                          batch_size=args.batchsize, 
                                          shuffle=False)


def MCDropout(act_vec, p=0.5, apply=True):
    return F.dropout(act_vec, p=p, training=apply)

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        MCDropout(x_reconst, apply=True)
        return x_reconst, mu, log_var

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training
if args.type == 'train' : 
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(data_loader):
            # Forward pass
            x = x.to(device).view(-1, image_size)
            x_reconst, mu, log_var = model(x)
            
            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                    .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
        
        with torch.no_grad():
            # Save the sampled images
            z = torch.randn(batch_size, z_dim).to(device)
            out = model.decode(z).view(-1, 1, 28, 28)
            save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

            # Save the reconstructed images
            out, _, _ = model(x)
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
    torch.save(model.state_dict(), '{}'.format(args.model))

elif args.type == 'test' :
    if os.path.isfile('{}'.format(args.model)) : 
        model.load_state_dict(torch.load('{}'.format(args.model)))
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(testdata_loader):
                print('[{}/{}]'.format(i, len(testdata_loader)))
                for j in range(test_trials) : 
                    if j == 0 : 
                        x = x.to(device).view(-1, image_size)
                        output = model(x.to(device))
                        output = output[0].view(args.batchsize, 28, 28).cpu()
                    else : 
                        x = x.to(device).view(-1, image_size)
                        output_ = model(x.to(device))
                        output_ = output_[0].view(args.batchsize, 28, 28).cpu()
                        output = tf.concat([output, output_], axis = 0)
                output = tf.nn.moments(output, axes=[0])
                result = tf.concat([output[1], x.view(28,28).cpu()], axis = 0).numpy()
                save_image(torch.from_numpy(result), os.path.join(sample_dir, '{}reconst-{}.png'.format((i+1),y)))

    else :
        print('need model.pth! so train a model first')
        exit()