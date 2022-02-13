from vhh_cmc.Configuration import Configuration
from vhh_cmc.PreProcessing import PreProcessing
import os
import numpy as np
import cv2

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix


class Evaluation(object):
    """
    Evaluation class includes all methods to evaluate the implemented algorithms.
    """

    def __init__(self, config_instance: Configuration):
        """
        Constructor.

        :param config_instance: object instance of type Configuration
        """
        print("create instance of class evaluation ... ")

        self.config_instance = config_instance
        self.path_eval_dataset = self.config_instance.path_eval_dataset
        self.path_gt_data = self.config_instance.path_gt_data
        self.path_eval_results = self.config_instance.path_eval_results

        self.pre_processing_instance = PreProcessing(self.config_instance)

        self.all_shot_file_list = []
        self.final_dataset_np = []

    def load_cmc_eval_db_v2(self):
        """
        This method is used to load the dataset used to evaluate the algorithm.
        The dataset must have the following structure:
        dataset_root_dir/training_data/
        dataset_root_dir/training_data/tilt/
        dataset_root_dir/training_data/pan/
        dataset_root_dir/training_data/annotation/xxx.flist
        """
        print("load eval dataset ... ")

        # load samples
        tilt_samples_path = self.path_eval_dataset + "/training_data/tilt/"
        pan_samples_path = self.path_eval_dataset + "/training_data/pan/"
        na_samples_path = self.path_eval_dataset + "/training_data/na/"
        path_annotations = self.path_eval_dataset + "/annotation_new/"

        tilt_shot_file_list = os.listdir(tilt_samples_path)
        pan_shot_file_list = os.listdir(pan_samples_path)
        na_shot_file_list = os.listdir(na_samples_path)

        self.all_shot_file_list = tilt_shot_file_list + pan_shot_file_list + na_shot_file_list
        self.all_shot_file_list.sort()
        all_shot_file_np = np.array(self.all_shot_file_list)
        #print(len(self.all_shot_file_list))

        # load groundtruth labels
        #test_gt_labels_file = path_annotations + "/annotations_tiny.csv"
        test_gt_labels_file = path_annotations + "/annotations.csv"
        #print(test_gt_labels_file)
      
        fp = open(test_gt_labels_file, 'r')
        lines = fp.readlines()
        fp.close()

        gt_annotation_list = []
        for line in lines:
            #print(line)
            line = line.replace('\n', '')
            line = line.replace('\\', '/')
            line = line.replace('\ufeff', '')
            line_split = line.split(';')
            gt_annotation_list.append([line_split[0], line_split[2], line_split[3]])
        gt_annotation_np = np.array(gt_annotation_list)

        final_dataset = []
        for i in range(0, len(gt_annotation_np)):
            #path = os.path.join(self.path_eval_dataset, gt_annotation_np[i][0])
            path = gt_annotation_np[i][0]
            video_name = path.split('/')[-1]
            start = int(gt_annotation_np[i][1])
            stop = int(gt_annotation_np[i][2]) - 1
            class_name = gt_annotation_np[i][0].split('/')[1]

            #print(video_name)
            idx = np.squeeze(np.where(video_name == all_shot_file_np)[0])
            sample_path = all_shot_file_np[idx]
            final_dataset.append([path, i, start, stop, class_name])

        #final_dataset.sort()
        self.final_dataset_np = np.array(final_dataset)
        print(self.final_dataset_np)

    def load_vhhmmsi_GT_V2_db(self):
        """
        This method is used to load the dataset used to evaluate the algorithm.
        The dataset must have the following structure:
        dataset_root_dir/training_data/
        dataset_root_dir/training_data/tilt/
        dataset_root_dir/training_data/pan/
        dataset_root_dir/training_data/annotation/xxx.flist
        """
        print("load eval dataset ... ")

        path_annotations = self.path_eval_dataset + "/annotations/cmc/"

        # load groundtruth labels
        test_gt_labels_file = os.listdir(path_annotations)
        test_gt_labels_file.sort()
        print(test_gt_labels_file)
        #test_gt_labels_file = test_gt_labels_file[1:2]
        #print(test_gt_labels_file)

        final_dataset = []
        gt_annotation_list = []
        for file in test_gt_labels_file:
            fp = open(path_annotations + file, 'r')
            lines = fp.readlines()
            fp.close()
            lines = lines[1:]
            
            for line in lines:
                #print(line)
                line = line.replace('\n', '')
                line = line.replace('\\', '/')
                line = line.replace('\ufeff', '')
                line_split = line.split(';')
                #print(line_split)
                #if(line_split[4] == "track"):
                #    #line_split[4] = "na"
                #    continue

                gt_annotation_list.append([line_split[0], int(line_split[1]), int(line_split[2]), int(line_split[3]), line_split[4]])
                
        gt_annotation_np = np.array(gt_annotation_list)
        self.final_dataset_np = gt_annotation_np

    def run_evaluation(self, idx=None):
        """
        This method is used to start and run the evaluation process.
        """
        print("calculate evaluation metrics ... ")

        if (idx != None):
            results_file_list = [f for f in os.listdir(self.path_eval_results) if f.endswith('.csv')][idx:idx+1]
            self.final_dataset_np = self.final_dataset_np[idx:idx+1]
            #print(self.final_dataset_np)
            #exit()
        else:
            # load all predictions and merge
            results_file_list = [f for f in os.listdir(self.path_eval_results) if f.endswith('.csv')]
            results_file_list.sort()
            print(results_file_list)

            #print(self.path_eval_results)
            #exit()

            final_gt_np = self.final_dataset_np
            print(self.final_dataset_np[:20])
            idx_track = np.where(self.final_dataset_np[:, 4:] == "track")[0]
            print(idx_track)
            #exit()

        pred_list = []
        pred_np = []
        for i, file in enumerate(results_file_list):
            results_file = os.path.join(self.path_eval_results, file)
            #print(results_file)
            fp = open(results_file, 'r')
            lines = fp.readlines()
            lines = lines[1:]
            lines.sort()
            fp.close()

            for line in lines:
                line = line.replace('\n', '')
                line_split = line.split(';')
                pred_list.append([line_split[0], int(line_split[1]), int(line_split[2]), int(line_split[3]), line_split[4]])
        pred_np = np.array(pred_list)

        # add sorting part
        pred_sort_np = pred_np[pred_np[:, 1].argsort()]
        final_dataset_sort_np = self.final_dataset_np[self.final_dataset_np[:, 1].argsort()]

        # remove paths
        for i in range(0, len(final_dataset_sort_np)):
            name = final_dataset_sort_np[i][0].split('/')[-1]
            final_dataset_sort_np[i][0] = name

        pred_np_prep = np.squeeze(pred_sort_np[:, 4:])
        gt_np_prep = np.squeeze(final_dataset_sort_np[:, 4:])

        accuracy, precision, recall, f1_score = self.calculate_metrics(y_score=pred_np_prep, y_test=gt_np_prep)
        return accuracy, precision, recall, f1_score

    def calculate_metrics(self, y_score, y_test):
        """
        This method is used to calculate the metrics: precision, recall, f1score.
        Furthermore, the confusion matrix is generated and stored as figure on a specified location.

        :param y_score: This parameter must hold a valid numpy array with the class prediction per shot .
        :param y_test: This parameter must hold a valid numpy array with the groundtruth labels per shot.
        """
        #print(len(y_score))
        #print(len(y_test))

        y_score = [s.lower() for s in y_score]
        y_test = [s.lower() for s in y_test]

        self.config_instance.class_names = [s.lower() for s in self.config_instance.class_names]

        #print(y_score)
        #print(y_test)
        print(self.config_instance.class_names)

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y_test, y_score)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_test, y_score, average='macro')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test, y_score, average='macro')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test, y_score, average='macro')
        print('F1 score: %f' % f1)

        # confusion matrix
        matrix = confusion_matrix(y_test, y_score, labels=self.config_instance.class_names)
        print(matrix)
        print(classification_report(y_test, y_score))

        print("save confusion matrix ...")
        self.plot_confusion_matrix(cm=matrix,
                                   target_names=self.config_instance.class_names,
                                   title='Confusion matrix',
                                   cmap=None,
                                   normalize=True,
                                   path=self.config_instance.path_eval_results + "/confusion_matrix_normalize.pdf")
        self.plot_confusion_matrix(cm=matrix,
                                   target_names=self.config_instance.class_names,
                                   title='Confusion matrix',
                                   cmap=None,
                                   normalize=False,
                                   path=self.config_instance.path_eval_results + "/confusion_matrix.pdf")
        ''''''

        return accuracy, precision, recall, f1

    def plot_confusion_matrix(self, cm=None,
                          target_names=[],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          path=""):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        plt.rc('pdf', fonttype=42)


        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure() #figsize=(8, 6)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Ground truth label')
        plt.xlabel('Predicted label - accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        # plt.savefig(path)
        plt.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.4)
        plt.savefig(path, dpi=500)

    def exportExperimentResults(self, fName, cmc_results_np: np.ndarray):
        """
        Method to export cmc results as csv file.

        :param fName: [required] name of result file.
        :param cmc_results_np: numpy array holding the camera movements classification predictions for each shot of a movie.
        """

        print("export results to csv!")

        if (len(cmc_results_np) == 0):
            print("ERROR: numpy is empty")
            exit()

        print(self.config_instance.path_eval_results)
        fp = open(self.config_instance.path_eval_results + "/" + fName.split('/')[-1].split('.')[0] + ".csv", 'w')

        header = "exp_name;mvi_mv_ratio;threshold_significance;threshold_consistency;mvi_window_size;region_window_size;active_threshold;acc;prec;rec;f1_score"
        fp.write(header + "\n")

        for i in range(0, len(cmc_results_np)):
            tmp_line = str(cmc_results_np[i][0])
            for c in range(1, len(cmc_results_np[i])):
                tmp_line = tmp_line + ";" + str(cmc_results_np[i][c])
            fp.write(tmp_line + "\n")

        fp.close()
