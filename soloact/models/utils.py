import pickle
import datetime
import matplotlib.pyplot as plt

def dump_pickle(obj, name):
    """
    Saves an object to a pickle file.

    args
        obj (object)
        name (str) path of the dumped file
    """
    name = name + '{0:%Y-%m-%d %H:%M}'.format(datetime.datetime.now())
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    """
    Loads a an object from a pickle file.

    args
        name (str) path to file to be loaded

    returns
        obj (object)
    """
    with open(name, 'rb') as handle:
        return pickle.load(handle)

# NOT IMPLEMENTED
def quick_plot():
    # plot the total loss, category loss, and color loss
    lossNames = ["loss", "category_output_loss", "color_output_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
    	# plot the loss for both the training and validation data
    	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    	ax[i].set_title(title)
    	ax[i].set_xlabel("Epoch #")
    	ax[i].set_ylabel("Loss")
    	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
    		label="val_" + l)
    	ax[i].legend()

    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plt.savefig("{}_losses.png".format(args["plot"]))
    plt.close()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
