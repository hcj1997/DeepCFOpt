import numpy as np
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dot, Lambda, multiply, Reshape, multiply, \
    Concatenate, Dropout,Multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse
import DMF
import MLP


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[512,256,128,64]',
                        help="MLP layers. Note that the first layer is the concatenation "
                             "of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--userlayers', nargs='?', default='[512, 64]',
                        help="Size of each user layer")
    parser.add_argument('--itemlayers', nargs='?', default='[1024, 64]',
                        help="Size of each item layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='sgd',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--dmf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for DMF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


def get_model(train, num_users, num_items, userlayers, itemlayers, layers):
    user_matrix = K.constant(getTrainMatrix(train))
    item_matrix = K.constant(getTrainMatrix(train).T)
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layer
    # user_rating= Lambda(lambda x: tf.gather(user_matrix, tf.to_int32(x)))(user_input)
    # item_rating = Lambda(lambda x: tf.gather(item_matrix, tf.to_int32(x)))(item_input)
    user_rating = Lambda(lambda x: tf.gather(user_matrix, x))(user_input)
    item_rating = Lambda(lambda x: tf.gather(item_matrix, x))(item_input)
    user_rating = Reshape((num_items,))(user_rating)
    item_rating = Reshape((num_users,))(item_rating)

    left = Dense(128, activation='relu')(user_rating)
    for i in range(10):
        left = Dropout(0.2)(left)
        left = Dense(128, activation='relu')(left)

    # 定义表示学习的right
    right = Dense(128, activation='relu')(item_rating)
    for i in range(10):
        right = Dropout(0.2)(right)
        right = Dense(128, activation='relu')(right)

    # 表示学习的融合
    repre = Multiply()([left, right])

    # 定义函数学习
    user = Dense(100, activation='relu')(user_rating)
    item = Dense(100, activation='relu')(item_rating)
    mlp = Concatenate()([user, item])
    for i in range(10):
        mlp = Dropout(0.2)(mlp)
        mlp = Dense(128, activation='relu')(mlp)

    # 将表示学习和函数学习结合起来
    fusion = Multiply()([repre, mlp])
    output = Dense(1, activation='sigmoid',
                   name="prediction")(fusion)
    model = Model(inputs=[user_input, item_input], outputs=output)
    model.summary()
    return model


def getTrainMatrix(train):
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.int32)
    for (u, i) in train.keys():
        train_matrix[u][i] = 1
    return train_matrix


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    print('path=', path)
    print('dataset=', dataset)
    userlayers = eval(args.userlayers)
    itemlayers = eval(args.itemlayers)
    layers = eval(args.layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs
    verbose = args.verbose
    dmf_pretrain = args.dmf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("DeepCF arguments: %s " % args)
    model_out_file = 'Pretrain/%s_CFNet_%d.h5' % (args.dataset, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(train, num_users, num_items, userlayers, itemlayers, layers)
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')

    # Check Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)

    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        print('获取数据：')
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        print('又一次开始了')
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best CFNet model is saved to %s" % model_out_file)

