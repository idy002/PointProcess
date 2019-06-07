import os
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


#
# the plot function is copied from [ https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html ]
#
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


class Logger:
    def __init__(self, dir, args):
        self.global_step = 0
        self.local_step = 0
        self.evaluation_id = 0

        run = 0
        while os.path.exists(f"{dir}/run{run}"):
            run += 1
        dir = f"{dir}/run{run}"
        self.dir = dir
        self.cm_dir = f'{dir}/cm_dir'
        os.makedirs(dir)
        os.makedirs(self.cm_dir)
        self.writer = SummaryWriter(log_dir=dir)
        with open(f'{dir}/args.json', 'w') as f:
            json.dump(args, f, indent=4)

    def log_new_epoch(self, epoch):
        print(f"Epoch {epoch}")
        self.local_step = 0

    def log_train(self, time_loss, event_loss, merged_loss):
        self.global_step += 1
        self.local_step += 1
        print(f"{self.local_step} {self.global_step} Time loss: {time_loss:8.3f} Event loss: {event_loss:8.3f} Merged loss: {merged_loss:8.3f}")
        self.writer.add_scalars('loss', {'time': time_loss, 'event': event_loss, 'merged': merged_loss}, global_step=self.global_step)

    def log_evaluation(self, evaluation_result, is_test):
        self.evaluation_id += 1
        target_time, predicted_time, target_events, predicted_events = evaluation_result

        main_target_events = []
        main_predicted_events = []
        sub_target_events = []
        sub_predicted_events = []
        for te, pe in zip(target_events, predicted_events):
            if te == 0:
                main_target_events.append(0)
                sub_target_events.append(0)
            else:
                main_target_events.append(1)
                sub_target_events.append(te - 1)
            if pe == 0:
                main_predicted_events.append(0)
                sub_predicted_events.append(0)
            else:
                main_predicted_events.append(1)
                sub_predicted_events.append(pe - 1)

        main_classes = np.array(['Ticket', 'Error'])
        plot_confusion_matrix(main_target_events, main_predicted_events, main_classes, normalize=True)
        plt.savefig(f'{self.cm_dir}/main_{self.evaluation_id}')
        plt.close()

        sub_classes = np.array(['PRT', 'CNG', 'IDC', 'COMM', 'LMTP', 'MISC'])
        plot_confusion_matrix(sub_target_events, sub_predicted_events, sub_classes, normalize=True)
        plt.savefig(f'{self.cm_dir}/sub_{self.evaluation_id}')
        plt.close()


        time_result = {
            'Time MAE': mean_absolute_error(target_time, predicted_time)
        }
        maintype_result = {
            'Accuracy': accuracy_score(main_target_events, main_predicted_events),
            'F1 score': f1_score(main_target_events, main_predicted_events, average='macro'),
            'Precision': precision_score(main_target_events, main_predicted_events, average='macro'),
            'Recall': recall_score(main_target_events, main_predicted_events, average='macro')
        }
        subtype_result = {
            'Accuracy': accuracy_score(sub_target_events, sub_predicted_events),
            'F1 score': f1_score(sub_target_events, sub_predicted_events, average='macro'),
            'Precision': precision_score(sub_target_events, sub_predicted_events, average='macro'),
            'Recall': recall_score(sub_target_events, sub_predicted_events, average='macro')
        }
        if is_test:
            print("")
            print("Test Set Evaluation:")
            print("")
        print("MainType " + "   ".join([f'{k:8}:{v:6.3f}' for k, v in maintype_result.items()]))
        print("SubType " + "   ".join([f'{k:8}:{v:6.3f}' for k, v in subtype_result.items()]))
        print("   ".join([f'{k:8}:{v:6.3f}' for k, v in time_result.items()]))
        self.writer.add_scalars('time evaluation', time_result, global_step=self.global_step)
        self.writer.add_scalars('maintype event evaluation', maintype_result, global_step=self.global_step)
        self.writer.add_scalars('subtype event evaluation', subtype_result, global_step=self.global_step)

