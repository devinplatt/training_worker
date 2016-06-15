import joblib
import os
import keras
from core import timeit


@timeit
def accuracy(predicted, real):
    num_right = 0
    for i in range(len(real)):
        if predicted[i]  == real[i]:
            num_right += 1
    return 'accuracy: {}'.format(num_right / float(len(real)))


@timeit
def get_evaluation_results_dictionary(model, x_test, y_test):
    predicted = model.predict_classes(x_test, batch_size=64)
    # proba = model.predict_proba(x_test, batch_size=64)
    stats = {"accuracy": accuracy(predicted,
                                  [vector.argmax() for vector in y_test])}
    # TODO: get per-label precision and recall values.
    return stats
