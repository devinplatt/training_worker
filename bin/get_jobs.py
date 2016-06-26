# Get message.
# Get all info from message.
# Download data.
# Prepare data for model (eg. using LabelMap)
# Load model with training parameters and compile and train.
# Prepare model file, evaluation results.
# Upload model file, evaluation results.
import json
import joblib
import os

import numpy as np

from lib.core import ensure_dirs_exist, timeit
from lib import train
from lib import evaluation

import boto
from boto import s3  # or import boto.s3
from boto.s3.key import Key

LABEL_SET_FID = '1'


# Stub function to create a message.
def get_message():
    model_archiecture_string = '{"layers": [{"cache_enabled": true, "dims": [1, 21, 40], "name": "Reshape", "input_shape": [840]}, {"b_constraint": null, "name": "Convolution2D", "subsample": [1, 1], "nb_col": 10, "activation": "linear", "W_constraint": null, "dim_ordering": "th", "input_shape": [1, 21, 40], "cache_enabled": true, "init": "glorot_uniform", "nb_filter": 8, "b_regularizer": null, "W_regularizer": null, "nb_row": 7, "activity_regularizer": null, "border_mode": "valid"}, {"cache_enabled": true, "activation": "relu", "name": "Activation"}, {"name": "MaxPooling2D", "strides": [2, 2], "dim_ordering": "th", "pool_size": [2, 2], "cache_enabled": true, "border_mode": "valid"}, {"cache_enabled": true, "name": "Dropout", "p": 0.25}, {"cache_enabled": true, "name": "Flatten"}, {"b_constraint": null, "name": "Dense", "activity_regularizer": null, "W_constraint": null, "cache_enabled": true, "init": "uniform", "activation": "linear", "input_dim": null, "b_regularizer": null, "W_regularizer": null, "output_dim": 32}, {"cache_enabled": true, "activation": "relu", "name": "Activation"}, {"cache_enabled": true, "name": "Dropout", "p": 0.25}, {"b_constraint": null, "name": "Dense", "activity_regularizer": null, "W_constraint": null, "cache_enabled": true, "init": "uniform", "activation": "linear", "input_dim": null, "b_regularizer": null, "W_regularizer": null, "output_dim": 5}, {"cache_enabled": true, "activation": "softmax", "name": "Activation"}], "name": "Sequential"}'
    training_parameters = '{"early_stopping": true, "batch_size": 32, "loss": "categorical_crossentropy", "optimizer": "Adagrad"}'
    mid = '1'
    xfid = '5'
    yfid = '1'
    dspid = '1'
    return {'model_archiecture': model_archiecture_string,
            'training_parameters': training_parameters,
            'mid': mid,
            'xfid': xfid,
            'yfid': yfid,
            'dspid': dspid}


# Stub function to get dsid.
def get_dsid_from_dspid(dspid):
    return '1'


# Stub function to get label map.
def get_label_map_from_dsid(dsid):
    lines = [line.strip() for line in open('label_map.txt').readlines()]
    label_map = {label: i for i, label in enumerate(lines)}
    return label_map
    # return json.load(open('label_map.txt'))


# Stub function to label set.
def get_label_set_from_dsid(dsid):
    return json.load(open('label_set.json'))


# Stub function to test set.
# Returns data point name for each data point in the test partition of the data
# set. Ie. these are the values before the file extensions for the feature
# files.
def get_test_set_from_dspid(dspid):
    return set(line.strip() for line in open('test_set.txt').readlines())


# Stub function to get mel features.
# TODO: test this.
@timeit
def get_mel_stub(dsid, data_dir):
    aws_region = 'us-west-1'
    s3_bucket_name = 'platt-data'
    s3_connection = boto.s3.connect_to_region(aws_region)
    bucket = s3_connection.get_bucket(s3_bucket_name)
    file_names = []
    key_names = []
    # TODO: use parse.get_feature_s3_prefix here instead of hard-coded string.
    for key in bucket.list('features/magnatagatune'):
        file_name = os.path.join(data_dir, key.name)
        ensure_dirs_exist([os.path.dirname(file_name)])
        try:
            key.get_contents_to_filename(file_name)
        except Exception as e:
            print(e)
            continue
        file_names.append(file_name)
        key_names.append(key.name.split('.')[0])
    return get_mel_stub_from_local(dsid, data_dir)
    #return {name: joblib.load(file_name) for name, file_name in zip(key_names,
    #                                                                file_names)}


# Function to load already downloaded features.
# Very dataset-specific stub.
@timeit
def get_mel_stub_from_local(dsid, data_dir):
    aws_region = 'us-west-1'
    s3_bucket_name = 'platt-data'
    s3_connection = boto.s3.connect_to_region(aws_region)
    bucket = s3_connection.get_bucket(s3_bucket_name)
    file_names = []
    key_names = []
    x = list(os.walk(data_dir))
    # 3 to 19 is "e" to "0", for Magnatagatune prefixes.
    for i in range(3,19):
        parent = x[i][0]
        for fname in x[i][2]:
            file_names.append(os.path.join(parent, fname))
            magna_prefix = parent.split('/')[-1]
            magna_suffix = fname.split('.')[0]
            key_name = os.path.join(magna_prefix, magna_suffix)
            key_names.append(key_name)
    return {name: joblib.load(file_name) for name, file_name in zip(key_names,
                                                                    file_names)}


# For now, we assume that all data can be loaded into memory,
# so download_features() loads features into memory, as dictionaries.
# TODO: use DSPID to figure out validation set.
def download_features(fid, dsid, data_dir):
    if fid == LABEL_SET_FID:
        return get_label_set_from_dsid(dsid)
    elif fid == '5':
        return get_mel_stub(dsid, data_dir)


# Function to load already downloaded features.
def download_features_from_local(fid, dsid, data_dir):
    if fid == LABEL_SET_FID:
        return get_label_set_from_dsid(dsid)
    elif fid == '5':
        return get_mel_stub_from_local(dsid, data_dir)


# TODO: time profile this on smaller datasets.
# TODO: looks like we fill the whole 12GB partition downloading the dataset,
# deal with this.
def get_features(xfid, yfid, dspid, data_dir):
    ensure_dirs_exist([data_dir])
    dsid = get_dsid_from_dspid(dspid)
    x_feats = download_features(xfid, dsid, data_dir)
    y_feats = download_features(yfid, dsid, data_dir)
    # Convert the dictionaries to parallel lists, using only keys that they have
    # in common.
    common_keys = list(set(x_feats.keys()) & set(y_feats.keys()))
    test_set = get_test_set_from_dspid(dspid)
    x_feats_train = [x_feats[key] for key in common_keys if key not in test_set]
    y_feats_train = [y_feats[key] for key in common_keys if key not in test_set]
    x_feats_test = [x_feats[key] for key in common_keys if key in test_set]
    y_feats_test = [y_feats[key] for key in common_keys if key in test_set]

    return x_feats_train, y_feats_train, x_feats_test, y_feats_test


# Function to load already downloaded features.
def get_features_from_local(xfid, yfid, dspid, data_dir):
    ensure_dirs_exist([data_dir])
    dsid = get_dsid_from_dspid(dspid)
    x_feats = download_features_from_local(xfid, dsid, data_dir)
    y_feats = download_features_from_local(yfid, dsid, data_dir)
    # Convert the dictionaries to parallel lists, using only keys that they have
    # in common.
    common_keys = list(set(x_feats.keys()) & set(y_feats.keys()))
    test_set = get_test_set_from_dspid(dspid)
    x_feats_train = [x_feats[key] for key in common_keys if key not in test_set]
    y_feats_train = [y_feats[key] for key in common_keys if key not in test_set]
    x_feats_test = [x_feats[key] for key in common_keys if key in test_set]
    y_feats_test = [y_feats[key] for key in common_keys if key in test_set]

    return x_feats_train, y_feats_train, x_feats_test, y_feats_test


# Take ((feature, metadata), targets) pairs and split them into
# (feature, target) pairs.
# (We also remove the feature metadata)
# Then turn label targets in 0/1 one vectors.
# Then split feature segments into separate examples.
def prepare_features_for_model(x_feats, y_feats, label_map):
    x_feats_new = []
    y_feats_new = []
    for x, y in zip(x_feats, y_feats):
        for label in y:
            x_feats_new.append(x[0])
            y_feats_new.append(label)
    y_feats_new = train.string_labels_to_binary_vectors(y_feats_new,
                                                        label_map)
    x_feats_new, y_feats_new = train.expand_features_and_labels(x_feats_new,
                                                                y_feats_new)
    return np.vstack(x_feats_new), np.vstack(y_feats_new)


def upload_results(mid, model, hist, eval_dict, results_dir = '/tmp/results/'):
    aws_region = 'us-west-1'
    s3_bucket_name = 'platt-data'
    s3_connection = boto.s3.connect_to_region(aws_region)
    bucket = s3_connection.get_bucket(s3_bucket_name)

    model_path = os.path.join(results_dir, 'model')
    save_model_to_path_stub(model, model_path)
    history_filename = os.path.join(results_dir, 'history.txt')
    eval_filename = os.path.join(results_dir, 'eval.json')
    with open(history_filename, 'w') as history_file:
        history_file.write(hist.history)
    json.dump(eval_dict, open(eval_filename, 'w'))

    for file_name in os.list_dir(results_dir):
        full_file_name = os.path.join(results_dir, file_name)
        s3_output_key = 'models/{}/{}'.format(mid, file_name)
        key = Key(s3Bucket)
        key.key = s3_output_key
        try:
            key.set_contents_from_filename(full_file_name)
        except Exception as e:
            print(e)

# Get message.
# Get all info from message.
# Download data.
# Prepare data for model (eg. using LabelMap)
# Load model with training parameters and compile and train.
# Prepare model file, evaluation results.
# Upload model file, evaluation results.

# For now, we assume that all data can be loaded into memory,
# so download_features() loads features into memory, as dictionaries.
def do_job(message):
    try:
        mid = message['mid']
        dspid = message['dspid']
        xfid = message['xfid']
        yfid = message['yfid'] 
        architecture_json_string = message['model_archiecture']
        training_parameters = json.loads(message['training_parameters'])
    except Exception as e:
        # TODO: log error here.
        print('Could not parse message.')
        print(e)

    # TODO: add support for non-label values.
    if yfid != LABEL_SET_FID:
        print('No support yet for target features which are not string labels.')
        return False

    # data_dir = '/tmp/datasets/'
    data_dir = '/dev/tmp/datasets'
    x_feats_train, y_feats_train, x_feats_test, y_feats_test = get_features(xfid,
                                                                            yfid,
                                                                            dspid,
                                                                            data_dir)
    # Now we are assumming yfid == LABEL_SET_FID
    dsid = get_dsid_from_dspid(dspid)
    label_map = get_label_map_from_dsid(dsid)
    X, y = prepare_features_for_model(x_feats_train,
                                      y_feats_train,
                                      label_map)
    X_test, y_test = prepare_features_for_model(x_feats_test,
                                                y_feats_test,
                                                label_map)

    model = train.load_model_from_architecture_string(architecture_json_string)
    train.compile_model(model, training_parameters)
    # TODO: use k-fold cross validaiton:
    # https://github.com/fchollet/keras/issues/1711
    # and set "validation_split" in training_parameters
    # training_parameters["validation_split"] = 0.1
    hist = train.train_model(model, X, y, training_parameters)
    eval_dict = evaluation.get_evaluation_results_dictionary(model,
                                                             X_test,
                                                             y_test)

    # Upload model architecture, model parameters, hist.history, eval results.
    upload_results(mid, model, hist, eval_dict)

    # TODO: clean up local files, unless going to use them on next job?
