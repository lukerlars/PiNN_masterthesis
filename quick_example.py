import torch 
import matplotlib.pyplot as plt
import numpy as np


class Neural_net(torch.nn.Module):
    def __init__(self):
        super(Neural_net, self).__init__()

        self.layer1 = torch.nn.Linear(1,20)
        self.tanh = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(20,1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x


class Simple_pinn():
    def __init__(self, epochs):
        self.model = Neural_net() # Simple 1-layer PyTorch neural network model
        self.some_xs = np.linspace(0,2,10)
        self.some_other_xs = torch.linspace(0,2,100,requires_grad=True).reshape(-1,1) 

        self.x_domain =torch.linspace(0,1,100,requires_grad=True).reshape(-1,1) 
        self.optimizer = torch.optim.LBFGS(params = self.model.parameters(), lr =0.001, max_iter=200)
        self.c0 = 1.0
        self.epochs = epochs
    
    def wrap_torch_grad(self, f,x):
        return torch.autograd.grad(f,x,
        grad_outputs=torch.ones_like(x),
        retain_graph=True,
        create_graph=True)[0]

    
    def de_loss(self):
        def coef(x):
            return (x + (1+3*x**2)/(1+x+x**3))
        def expr(x):
            return x**3 + 2*x + x**2*((1+3*x**2)/(1+x+x**3))
        
        pred = self.model(self.x_domain)
        dpred = self.wrap_torch_grad(pred, self.x_domain) 
        
        z0 = torch.mean((dpred + coef(self.x_domain)*pred -expr(self.x_domain))**2)
        ic = (self.c0 - pred[0])**2
        
        return z0 + ic
    
    def true_sol(self,x):
        return x**2 + np.exp(-x**2/2)/(1+x+x**3)

    
    def train(self):
        self.model.train()    
        for epoch in range(self.epochs):
            def closure():
                self.optimizer.zero_grad()
                loss = self.de_loss()
                loss.backward()
                return loss
            self.optimizer.step(closure=closure)
            print(self.de_loss())
        plt.plot(self.some_other_xs.detach(), self.model(self.some_other_xs).detach(), label = 'pred')
        plt.scatter(self.some_xs, [self.true_sol(x) for x in self.some_xs], label = 'analytic')
        plt.legend()
        plt.grid()
        plt.show()
            

if __name__ == '__main__':
    instance = Simple_pinn(epochs= 10)
    instance.train()