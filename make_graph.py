import numpy as np
import matplotlib.pyplot as plt

def draw_loss_graph(train_losses, test_losses):

    # total loss
    fig = plt.figure()
    plt.ylim(-0.01, 1)
    plt.plot(train_losses,label='train')
    plt.plot(test_losses,label='test')
    plt.title('loss')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    fig.savefig("loss.png")

def draw_acc_graph(train_accs,test_accs):

    # class accuracy
    fig = plt.figure()
    plt.ylim(-0.01, 1)
    plt.plot(train_accs,label='train')
    plt.plot(test_accs,label='test')
    plt.title('accuracy')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    fig.savefig("accuracy.png")
