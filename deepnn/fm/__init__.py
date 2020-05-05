# import matplotlib.pyplot as plt
# import tensorflow as tf
# from pandas import np
# import pandas as pd
#
# # TensorFlow2.0 实现FM
# # https://blog.csdn.net/qq_34333481/article/details/103919923
#
# from deepnn.fm.FM import FM_Model
#
# feat_dict = {}
# tc = 1
#  # Continuous features
# for idx in continuous_range_:
#     feat_dict[idx] = tc
#     tc += 1
#
# features = line.rstrip('\n').split('\t')
#  #for idx in categorical_range_:
#       if features[idx] == '' or features[idx] not in dis_feat_set:
#          continue
#       if features[idx] not in cnt_feat_set:
#           cnt_feat_set.add(features[idx])
#           feat_dict[features[idx]] = tc
#           tc += 1
#
# train_batch_dataset = get_batch_dataset(train_label_path, train_idx_path, train_value_path)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#
# output = FM_Model(idx, value)
# loss = cross_entropy_loss(y_true=label, y_pred=output)
#
# reg_loss = []
# for p in model.trainable_variables:
#     reg_loss.append(tf.nn.l2_loss(p))
# reg_loss = tf.reduce_sum(tf.stack(reg_loss))
# loss = loss + model.reg_l2 * reg_loss