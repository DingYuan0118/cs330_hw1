#plot the loss
import numpy as np
import matplotlib.pyplot as plt

def loss_plot(N, K, path = ".\\mylog", keyword = "train"):
    """
    
    Args:
        N: means N-way
        K: means K-shot 
        path: the path of log file

    Returns:
        A figure that plot the train or test loss or acc
    """
    if keyword == "train":
        loss = np.load(path + "{}-way-{}-shot_TrainLoss.npz".format(N, K))
    elif keyword == 'test':
        loss = np.load(path + "{}-way-{}-shot_TestLoss.npz".format(N, K))
    elif keyword == "acc" :
        acc = np.load(path + "{}-way-{}-shot_acc.npz".format(N, K))
    else:
        raise ValueError("keyword must be 'train' or 'test' or 'acc'")

    step = np.load(path + "{}-way-{}-shot_step.npz".format(N, K))
    fig, ax = plt.subplots()
    if keyword in ["train", "test"]:
        ax.plot(step, loss, label = "{} loss".format(keyword))
        ax.set_xlabel('step')
        ax.set_xlabel("{} loss".format(keyword))
        fig.save("{}_loss_of_{}-way-{}-shot".format(keyword, N, K))
    else:
        ax.plot(step, acc, label = "Accuracy")
        ax.set_xlabel('step')
        ax.set_xlabel("{}".format(keyword))
        fig.save("{}_of_{}-way-{}-shot".format(keyword, N, K))

