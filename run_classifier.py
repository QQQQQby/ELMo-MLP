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

        self.label_size = len(args.labels)
        # 预处理数据
        self.data_train = self._read_xml(os.path.join(args.dataset_path, 'answers_train.xml'))
        self.data_dev = self._read_xml(os.path.join(args.dataset_path, 'answers_dev.xml'))
        self.data_test = self._read_xml(os.path.join(args.dataset_path, 'answers_test.xml'))

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

        self.elmo = hub.Module("https://tfhub.dev/google/elmo/3")
        with tf.name_scope('labeled_text'):
            self.label_input = tf.placeholder(tf.int8, [None, self.label_size], name='labels')
            self.text_input = tf.placeholder(tf.string, [None], name='texts')
        self.embeddings = self.elmo(
            self.text_input,
            signature="default",
            as_dict=True)["default"]

        with tf.name_scope('ELMo_Classifier'):
            self.h_size = int(self.embeddings.shape[-1])  # embedding维度

            self.W = tf.Variable(tf.truncated_normal([self.h_size, self.label_size]), name='Weights')
            self.B = tf.Variable(tf.truncated_normal([self.label_size]), name='Bias')
            self.Z = tf.matmul(self.embeddings, self.W) + self.B

            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 tf.contrib.layers.l2_regularizer(self.args.lamb)(self.W))  # 使用L2正则化，防止过拟合

            self.true_label = tf.argmax(self.label_input, 1)
            self.prob = tf.nn.softmax(self.Z)
            self.pred = tf.argmax(self.prob, 1)
            self.correct = tf.equal(self.true_label, self.pred)
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
        eval_result = 'Iteration\tAccuracy\n'
        num_iterations = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.args.num_epochs):
                # 训练
                if self.args.do_train:
                    print('-' * 20 + 'Training epoch %d' % i + '-' * 20)
                    random.shuffle(self.data_train)
                    for start in range(0, len(self.data_train), self.args.batch_size):
                        num_iterations += 1
                        end = min(start + self.args.batch_size, len(self.data_train))
                        _, loss = sess.run([self.op, self.loss],
                                           feed_dict={
                                               self.text_input: [batch[0] for batch in self.data_train[start:end]],
                                               self.label_input: [batch[1] for batch in self.data_train[start:end]]})
                        print('batch[%d:%d]: %f' % (start, end, loss))

                # 计算dev集准确率
                if self.args.do_eval:
                    print('-' * 20 + 'evaluating current epoch' + '-' * 20)
                    c = 0
                    for start in range(0, len(self.data_dev), self.args.batch_size):
                        end = min(start + self.args.batch_size, len(self.data_dev))
                        c += sum(sess.run(self.correct,
                                          feed_dict={
                                              self.text_input: [batch[0] for batch in self.data_dev[start:end]],
                                              self.label_input: [batch[1] for batch in self.data_dev[start:end]]}))
                    acc = c / len(self.data_dev)
                    eval_result += str(num_iterations) + '\t' + str(acc) + '\n'
                    print('Accuracy:' + str(acc))

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

        with open(os.path.join(self.args.output_path, 'eval_' + str(num_iterations) + '.txt'), 'w') as f:
            f.write(eval_result)

    def labels_2_one_hot(self, data):
        for i in range(len(data)):
            k = -1
            for j in range(self.label_size):
                if data[i][1] == self.args.labels[j]:
                    k = j
                    break
            data[i][1] = [0 for i in range(self.label_size)]
            data[i][1][k] = 1

    @classmethod
    def _read_xml(cls, input_file):
        f = xml.dom.minidom.parse(input_file)
        nodes = f.getElementsByTagName('Thread')
        data = []
        l = len(nodes)
        for i in range(l):
            question = nodes[i].getElementsByTagName('RelQuestion')[0].getAttribute('RELQ_CATEGORY') + ' # '
            if nodes[i].getElementsByTagName('RelQBody')[0].firstChild:
                question += (nodes[i].getElementsByTagName('RelQSubject')[0].firstChild.data + ' ## ' +
                             cls._clean_text(nodes[i].getElementsByTagName('RelQBody')[0].firstChild.data))
            else:
                question += cls._clean_text(nodes[i].getElementsByTagName('RelQSubject')[0].firstChild.data)
            answers = []
            for answer in nodes[i].getElementsByTagName('RelCText'):
                answers.append(cls._clean_text(answer.firstChild.data))
            if len(answers) == 0:
                continue
            labels = []
            for label in nodes[i].getElementsByTagName('RelComment'):
                labels.append(label.getAttribute('RELC_FACT_LABEL'))
            for j in range(len(answers)):
                data.append([question + " ### " + answers[j], labels[j]])
        return data

    @classmethod
    def _clean_text(cls, text):  # 清洗数据
        newText = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
        newText = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", newText)
        newText = re.sub(r"\'s", " \'s", newText)
        newText = re.sub(r"\'ve", " \'ve", newText)
        newText = re.sub(r"n\'t", " n\'t", newText)
        newText = re.sub(r"\'re", " \'re", newText)
        newText = re.sub(r"\'d", " \'d", newText)
        newText = re.sub(r"\'ll", " \'ll", newText)
        newText = re.sub(r",", " , ", newText)
        newText = re.sub(r"!", " ! ", newText)
        newText = re.sub(r"\(", " \( ", newText)
        newText = re.sub(r"\)", " \) ", newText)
        newText = re.sub(r"\?", " \? ", newText)
        newText = re.sub(r"\s{2,}", " ", newText)
        return newText


def parse_args():  # 定义参数
    parser = argparse.ArgumentParser(description='Run ELMo with Tensorflow Hub.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
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
