import numpy as np
import pylab
from sklearn.metrics import confusion_matrix
import itertools



class Plotter:

    def __init__(self, no_epochs, label, extension, file_name):
        self.epoch = np.arange(no_epochs)
        self.ext = extension
        self.label = label
        self.file_name = file_name
        self.create_file()

    def plot_graph(self, y, y_label):
        pylab.figure(figsize=(15, 5))
        pylab.plot(range(len(y)), y, label=self.label)
        pylab.xlabel('epochs')
        pylab.ylabel(y_label)
        pylab.legend()
        pylab.savefig('./figures/' + self.file_name + '_' + y_label + '_' + self.ext)
        pylab.ion()
        pylab.show()
        self.write_to_file(y, y_label)

    def plot_confusion_matrix(self, y_test, y_pred, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=pylab.cm.coolwarm):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        cm = confusion_matrix(y_test, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        pylab.figure(figsize=(6.4, 4.8))
        pylab.imshow(cm, interpolation='nearest', cmap=cmap)
        pylab.title(title)
        pylab.colorbar()
        tick_marks = np.arange(len(classes))
        pylab.xticks(tick_marks, classes, rotation=45)
        pylab.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            pylab.text(j, i, format(cm[i, j], fmt),
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")

        pylab.ylabel('True label')
        pylab.xlabel('Predicted label')
        pylab.tight_layout()
        pylab.savefig('./figures/' + self.file_name + '_confusion_matrix_' + self.ext)
        pylab.ion()
        pylab.show()
        self.write_to_file(y_test, 'y_label')
        self.write_to_file(y_pred, 'y_pred')

    def plot_bar_chart(self, y_labels, x):
        x = x[::-1]
        y_pos = np.arange(len(y_labels))
        y_labels = y_labels[::-1]

        pylab.figure(figsize=(15, 5))
        pylab.barh(y_pos, x, align='center')
        pylab.yticks(y_pos, y_labels)
        for i, v in enumerate(x):
            pylab.text(v + 0.005, i, str(round(v, 3)), color='black', fontweight='bold')

        pylab.xlabel(self.label)
        pylab.title(self.file_name)
        pylab.savefig('./figures/' + self.file_name + '_bar_chart' + self.ext)
        pylab.show()

    def create_file(self):
        f = open('Project_' + self.file_name + 'Training_Report.py', 'w')
        f.write('from numpy import array\nfrom numpy import int64\n\n')
        f.close()

    def write_to_file(self, data, label):
        f = open('Project_'+self.file_name + 'Training_Report.py', 'a')
        f.write(label + '= ' + repr(data) + '\n')
        f.close()
