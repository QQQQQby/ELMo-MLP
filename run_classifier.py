# coding:utf-8

import argparse
import random
import os
import re
import xml.dom.minidom
import tensorflow_hub as hub
import tensorflow as tf


class ELMo:

    def __init__(self, args):
        self.args = args

        self.num_labels = len(args.labels)
        # 读取并预处理数据
        self.data_train = self._read_data(os.path.join(args.dataset_path, 'answers_train.txt'))
        self.data_dev = self._read_data(os.path.join(args.dataset_path, 'answers_dev.txt'))
        self.data_test = self._read_data(os.path.join(args.dataset_path, 'answers_test.txt'))

        for data in [self.data_train, self.data_dev, self.data_test]:
            self.labels_2_one_hot(data)

        print('Number of training data:', len(self.data_train))
        print('Number of data for evaluation:', len(self.data_dev))
        print('Number of data for testing:', len(self.data_test))
        print('data_train[0:3] =')
        for i in range(3):
            print(self.data_train[i])
        print('data_dev[0:3] =')
        for i in range(3):
            print(self.data_dev[i])
        print('data_test[0:3] =')
        for i in range(3):
            print(self.data_test[i])

        with tf.name_scope('labeled_text'):
            self.label_input = tf.placeholder(tf.int8, [None, self.num_labels], name='labels')
            self.text_input = tf.placeholder(tf.string, [None], name='texts')
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/3")
        self.embeddings = self.elmo(
            self.text_input,
            signature="default",
            as_dict=True)["default"]

        with tf.name_scope('ELMo_Classifier'):
            self.h_size = int(self.embeddings.shape[-1])  # embedding维度

            self.W = tf.Variable(tf.truncated_normal([self.h_size, self.num_labels]), name='Weights')
            self.B = tf.Variable(tf.truncated_normal([self.num_labels]), name='Bias')
            self.Z = tf.matmul(self.embeddings, self.W) + self.B

            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 tf.contrib.layers.l2_regularizer(self.args.lamb)(self.W))  # 使用L2正则化，防止过拟合

            self.prob = tf.nn.softmax(self.Z)
            self.pred_label = tf.argmax(self.prob, 1)
            self.true_label = tf.argmax(self.label_input, 1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.Z, labels=self.label_input)) + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.op = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

        # writer = tf.summary.FileWriter('./log/', tf.get_default_graph())
        # writer.close()

    def run(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print('Start running ELMo Classifier.')
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        print('-' * 20 + 'training' + '-' * 20)
        print('Number of epochs:', self.args.num_epochs)
        print('Batch Size:', self.args.batch_size)
        eval_result = ''
        num_iterations = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.args.num_epochs):
                # 训练
                if self.args.do_train:
                    print('-' * 20 + 'Training epoch %d' % epoch + '-' * 20)
                    random.shuffle(self.data_train)
                    for start in range(0, len(self.data_train), self.args.batch_size):
                        num_iterations += 1
                        end = min(start + self.args.batch_size, len(self.data_train))
                        _, loss = sess.run([self.op, self.loss],
                                           feed_dict={
                                               self.text_input: [batch[0] for batch in self.data_train[start:end]],
                                               self.label_input: [batch[1] for batch in self.data_train[start:end]]})
                        print('batch[%d:%d]: %f' % (start, end, loss))

                # 计算模型在dev集上的各项指标
                if self.args.do_eval:
                    print('-' * 20 + 'evaluating current epoch' + '-' * 20)
                    confusion_matrix = [[0 for j in range(self.num_labels)] for i in range(self.num_labels)]
                    for start in range(0, len(self.data_dev), self.args.batch_size):
                        end = min(start + self.args.batch_size, len(self.data_dev))
                        pred_label, true_label = sess.run(
                            [self.pred_label, self.true_label],
                            feed_dict={self.text_input: [batch[0] for batch in self.data_dev[start:end]],
                                       self.label_input: [batch[1] for batch in self.data_dev[start:end]]})
                        for i in range(len(true_label)):
                            confusion_matrix[pred_label[i]][true_label[i]] += 1

                    accuracy = self.get_accuracy(confusion_matrix)
                    precision_list, recall_list = self.get_precision_and_recall_list(confusion_matrix)
                    f1_score_list = [2 * p * r / (p + r) for p, r in zip(precision_list, recall_list)]

                    eval_result += 'Iteration ' + str(num_iterations) + ':\n'
                    eval_result += 'Accuracy: ' + str(accuracy) + '\n'
                    eval_result += 'Precision: ' + str(precision_list) + '\n'
                    eval_result += 'Recall: ' + str(recall_list) + '\n'
                    eval_result += 'F1 Score: ' + str(f1_score_list) + '\n'
                    eval_result += '\n'

                    print('Confusion Matrix:')
                    for i in range(self.num_labels):
                        print(confusion_matrix[i])
                    print('Accuracy:', accuracy)
                    print('Precision:', precision_list)
                    print('Recall:', recall_list)
                    print('F1 Score:', f1_score_list)

                # 计算test集中每一条数据对label的选择概率
                if self.args.do_test:
                    print('-' * 20 + 'testing current epoch' + '-' * 20)
                    test_result = ''
                    for start in range(0, len(self.data_test), self.args.batch_size):
                        end = min(start + self.args.batch_size, len(self.data_test))
                        t = sess.run(self.prob,
                                     feed_dict={
                                         self.text_input: [batch[0] for batch in self.data_test[start:end]],
                                         self.label_input: [batch[1] for batch in self.data_test[start:end]]})
                        test_result += '\n'.join(['\t'.join([str(j) for j in i]) for i in t]) + '\n'
                    with open(os.path.join(self.args.output_path, 'test_' + str(num_iterations) + '.tsv'), 'w') as f:
                        f.write(test_result)
                print()

        if self.args.do_eval:
            with open(os.path.join(self.args.output_path, 'eval_' + str(num_iterations) + '.txt'), 'w') as f:
                f.write(eval_result)

    # 将data的label转换为one hot形式
    def labels_2_one_hot(self, data):
        for i in range(len(data)):
            k = -1
            for j in range(self.num_labels):
                if data[i][1] == self.args.labels[j]:
                    k = j
                    break
            data[i][1] = [0 for i in range(self.num_labels)]
            data[i][1][k] = 1

    # 根据混淆矩阵计算准确率
    def get_accuracy(self, confusion_matrix):
        try:
            accuracy = sum([confusion_matrix[i][i] for i in range(self.num_labels)]) / sum(map(sum, confusion_matrix))
        except ZeroDivisionError:
            accuracy = 1.0
        return accuracy

    # 根据混淆矩阵计算查准率和查全率，其维度为[label数量]
    def get_precision_and_recall_list(self, confusion_matrix):
        precision = []
        recall = []
        for i in range(self.num_labels):
            try:
                precision.append(confusion_matrix[i][i] / sum([line[i] for line in confusion_matrix]))
            except ZeroDivisionError:
                precision.append(1.0)
            try:
                recall.append(confusion_matrix[i][i] / sum(confusion_matrix[i]))
            except ZeroDivisionError:
                recall.append(1.0)
        return precision, recall

    # 读取数据
    @classmethod
    def _read_data(cls, input_file):
        with open(input_file, 'r') as f:
            data = f.read()
        data = [d.split('\t') for d in data.split('\n')]
        for i in range(len(data)):
            data[i][0] += (' ### ' + data[i][1])
            data[i][1] = data[i][2]
            del data[i][2]
        return data


# 定义参数
def parse_args():
    parser = argparse.ArgumentParser(description='Run ELMo with Tensorflow Hub.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=1e-3,
                        help='Regularization coefficient for weights.')
    parser.add_argument('--output_path', type=str, default='results/d/',
                        help='Save path.')
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='Data path.')
    parser.add_argument('--labels', type=list, default=['True', 'False', 'NonFactual'],
                        help='Labels of texts.')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether to use gpu.')
    parser.add_argument('--do_train', type=bool, default=True,
                        help='whether to train the model.')
    parser.add_argument('--do_eval', type=bool, default=True,
                        help='whether to do the evaluation.')
    parser.add_argument('--do_test', type=bool, default=True,
                        help='whether to predict test data.')
    return parser.parse_args()


if __name__ == '__main__':
    model = ELMo(parse_args())
    model.run()
