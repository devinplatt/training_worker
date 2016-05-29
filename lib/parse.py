# Stub tables for lack of database.
# Note that the feature_type field is used as a file extension.
features = [{'FID': '1', 'feature_type': 'label'},
            {'FID': '2', 'feature_type': 'labelmap'},
            {'FID': '3', 'feature_type': 'mp3'},
            {'FID': '4', 'feature_type': 'mel',
             'bands': '40', 'fft': '2048', 'hop': '512', 'sr': '22050'},
            {'FID': '5', 'feature_type': 'stacked_mel',
             'approximate_window_length_in_ms': '500'}]
datasets =[{'DSID': '1', 'XFID': '5', 'YFID': '1'}]
dataset_partitions =[{'DSPID': '1', 'DSID': '1',
                       # Parameters determining cross-validation sets.
                      }]
models = [{'MID': '1', 'DSID': '1',
           'layers': [{},  # TODO: finish this.
                     ]}]


def get_feature_s3_prefix(DSID, FID):
    return 'features/{FID}/{DSID}/'.format(FID=FID, DSID=DSID)


def get_feature_s3_key_target(DSID, FID, FEATURE_TYPE):
    return get_feature_s3_prefix(DSID, FID) + '{object}.{FEATURE_TYPE}'.format(
                                                  object='{object}',
                                                  FEATURE_TYPE=FEATURE_TYPE)


def get_label_map(DSID):
    return 'labelmaps/{DSID}/labelmap.txt'.format(DSID=DSID)


def get_object_keys(FID, DSID):
    """
        Get the keys for available objects in S3 for the given DSID and FID.
    """
    pass


def get_common_object_keys(XFID, YFID, DSID):
    """
        return objects keys which are in both XFID and YFID.
    """
    return list(set(get_object_keys(XFID, DSID)) & set(
                                                       get_object_keys(
                                                           YFID, DSID)))
