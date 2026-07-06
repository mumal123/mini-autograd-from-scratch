from engine import Value
from nn import MLP,Neuron,Layer

n=MLP(3,[4,4,1]) #meaning 3 inputs, 4 neurons in first layer, 4 neurons in second layer, and 1 output neuron

xs=[[2.0,3.0,-1.0],
    [3.0,-1.0,5.0],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]] #sample input data


ys=[1.0,-1.0,-1.0,1.0]  #expected output data

for k in range(20):

    #forward pass
    ypred=[n(x) for x in xs]
    loss=sum((yout-ygt)**2 for ygt,yout in zip(ys,ypred))

    #backward pass
    for p in n.parameters():
        p.grad=0.0
    loss.backward()

    #update
    for p in n.parameters():
        p.data+=-0.01 *p.grad
    
    print(k,loss.data)
