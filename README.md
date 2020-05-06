# ELMo Based Classifier

## ELMo Introduction

**ELMo**, or **E**mbeddings from **L**anguage **Mo**del, is a bidirectional language model based on RNN.

Bidirectional LSTM units are **pre-trained** with unlabeled data to generate ELMo representations. Then we can easily **fine-tune** the model to help us to classify labeled texts.

In this repository, we use the pre-trained ELMo with [Tensorflow Hub](https://tfhub.dev/google/elmo/3).

**MLP**(**M**ulti-**L**ayer **P**erceptron) is used in our classification task. Accuracy might be higher if we used DNN or CNN rather than MLP.

## Dataset

The classification task is subtask B of SemEval-2019 Task 8: Fact Checking in Community Question Answering Forums. The data for this task is provided in another repository: https://github.com/tsvm/factcheck-cqa.

## Requirements

You should install python version 3.5 or later, Tensorflow version 1.x and Tensorflow Hub.

```
pip install tensorflow tensorflow-hub
```

## Run

```
optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs.
  --batch_size BATCH_SIZE
                        Batch size.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --lamb LAMB           Regularization coefficient for weights.
  --output_path OUTPUT_PATH
                        Save path.
  --dataset_path DATASET_PATH
                        Data path.
  --labels LABELS       Labels of texts.
  --use_gpu USE_GPU     whether to use gpu.
```

Train and evaluate:

```
python run_classifier.py
```

For Windows users:

```
./run.bat
```