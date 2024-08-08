from yacs.config import CfgNode as CN #固定用法
import os

_C = CN()

_C.DATASETS = CN()
_C.DATASETS.knearestvocabs = 'beijing-minfreq-5-vocab-dist-cell50.h5'
_C.DATASETS.timeknearestvocabs = 'beijing-vocab-timedist-cell10.h5'
_C.DATASETS.dataset = 'simplified_st_tdrive_4'
_C.DATASETS.grid_file = 'data/grid_instance_file'
_C.DATASETS.timeinterval_file = 'data/timeinterval_instance_file'
_C.DATASETS.textvocab_file = 'textvocab_instance_poi_len200'
_C.DATASETS.textembeddings_file = 'st_tdrive_texts_embeddings'
_C.DATASETS.poi_st_tdrive_test_similar_file = 'st_tdrive_poi_test_mostsimilar'
_C.DATASETS.poi_st_tdrive_vali_similar_file = 'st_tdrive_poi_vali_mostsimilar'

_C.DATASETS.poi_st_tdrive_test_knn_query = 'st_tdrive_poi_test_knn_query'
_C.DATASETS.poi_st_tdrive_test_knn_changedquery = 'st_tdrive_poi_test_knn_changedquery'
_C.DATASETS.poi_st_tdrive_test_knn_db = 'st_tdrive_poi_test_knn_db'
_C.DATASETS.poi_st_tdrive_vali_knn_query = 'st_tdrive_poi_vali_knn_query'
_C.DATASETS.poi_st_tdrive_vali_knn_changedquery = 'st_tdrive_poi_vali_knn_changedquery'
_C.DATASETS.poi_st_tdrive_vali_knn_db  = 'st_tdrive_poi_vali_knn_db'
# 'traj_tdrive_eval_100' #'traj_tdrive_eval'
_C.DATASETS.poi_st_tdrive_test_cluster = 'st_tdrive_poi_test_cluster'
_C.DATASETS.poi_st_tdrive_test_cluster_changed = 'st_tdrive_poi_test_cluster_changed'

_C.DATASETS.poi_st_tdrive_trajvecs = 'poi_st_tdrive_trajvecs_1w'
_C.DATASETS.poi_st_tdrive_starttime_embs = 'poi_st_tdrive_starttime_embs_1w'
_C.DATASETS.poi_st_tdrive_startloc_embs = 'poi_st_tdrive_startloc_embs_1w'
_C.DATASETS.poi_st_tdrive_endloc_embs = 'poi_st_tdrive_endloc_embs_1w'
_C.DATASETS.poi_st_tdrive_traveltime = 'poi_st_tdrive_traveltime_inputdata'
_C.DATASETS.poi_st_tdrive_traveltime_labels = 'st_tdrive_traveltime_labels'
_C.DATASETS.poi_st_tdrive_traveltime_distort = 'poi_st_tdrive_traveltime_inputdata_distort'
_C.DATASETS.poi_st_travel_distort_traj = 'st_travel_distort_traj'


_C.DATASETS.VAL = ('traj_tdrive_eval_1000', 'time_tdrive_eval_1000',
                   'testtraj_augment_tdrive_eval_2000', 'testtimes_augment_tdrive_eval_2000')
_C.DATASETS.TRAIN = ('traj_tdrive_train', 'time_tdrive_train',
                     'testtraj_augment_tdrive_train', 'testtimes_augment_tdrive_train')
_C.DATASETS.TEST = ('traj_tdrive_test', 'time_tdrive_test', 'testtraj_augment_tdrive_test', 'testtimes_augment_tdrive_test')

_C.DATASETS.val_frechet_dist_truth_file = 'frechet_traj_stdist_tdrive_eval.npy'
_C.DATASETS.test_frechet_dist_truth_file = 'frechet_traj_stdist_tdrive_test.npy'
_C.DATASETS.ori_val_trajs_file = 'ori_eval_trajs_100'
_C.DATASETS.ori_val_times_file = 'ori_eval_times_100'
_C.DATASETS.test_newsimi_raw_filepath = 'test_newsimi_raw'
_C.DATASETS.time2vec_file = 'time2vec_list'
_C.DATASETS.validata_mostsimilar_filepath = 'poi_validate_mostsimilar'
_C.DATASETS.vali_dataset = 'vali_positivetrajs'



_C.max_len = 150

_C.INPUT = CN()

_C.SOLVER = CN()
_C.SOLVER.eta1 = 1 #0.3
_C.SOLVER.eta2 = 1#0.3
_C.SOLVER.eta3 = 0.4
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY=0.0001
_C.SOLVER.LR_STEP = 10
_C.SOLVER.LR_GAMMA= 0.5
_C.SOLVER.val_batchsize = 64
_C.SOLVER.train_batchsize = 64
_C.SOLVER.test_batchsize = 32
_C.SOLVER.print_freq = 50
_C.SOLVER.save_freq = 2
_C.SOLVER.use_gpu = True
_C.SOLVER.epochs = 50

_C.MISC = CN()

_C.min_lon = 0.0
_C.min_lat = 0.0
_C.max_lon = 0.0
_C.max_lat = 0.0
_C.max_traj_len = 200
_C.min_traj_len = 20
_C.cell_size = 50
_C.cellspace_buffer = 500.0

_C.spatial_embedding_file= 'spatial_embedding_checpoints'

# ===========TrajCL=============
_C.feature_size = 64   # node2vec feature size
_C.embedding_size = 64  # GCN embedding size
_C.time2vec_size = 64  # date2vec output size
_C.hidden_size = 128  # LSTM hidden size
_C.num_layers = 2    # Spatial LSTM layer
_C.dropout_rate = 0
_C.concat = False  # whether concat or pairwise add of two embeddings


_C.early_stop = 30
_C.de_embedding_size = 64
_C.temperature = 0.05
_C.n_direction = 2
_C.vocab_size = 51182
_C.timevocab_size = 52605
_C.textvocab_size = 255

_C.spatial_embedding_size = 128  # GCN embedding size
_C.time_embedding_size = 256
_C.text_embedding_size = 256
_C.fusion_embedding_size = 512
_C.cell_embedding_dim = 128
_C.seq_embedding_dim = 256
_C.moco_proj_dim = _C.seq_embedding_dim // 2
_C.moco_nqueue = 2048
_C.moco_temperature = 0.07  # 0.05
_C.traveltime_embedding_size = 640

_C.trajcl_aug1 = 'mask'
_C.trajcl_aug2 = 'subset'
_C.local_mask_sidelen = _C.cell_size * 11

_C.trans_attention_head = 4
_C.trans_attention_dropout = 0.1
_C.trans_attention_layer = 2
_C.trans_pos_encoder_dropout = 0.1
_C.trans_hidden_dim = 2048

_C.traj_simp_dist = 100
_C.traj_shift_dist = 200
_C.traj_mask_ratio = 0.3
_C.traj_add_ratio = 0.3
_C.traj_subset_ratio = 0.7  # preserved ratio

_C.test_exp1_lcss_edr_epsilon = 0.25  # normalized

_C.dist_decay_speed = 0.8

# ===========Similarity=============
_C.distance_type = 'frechet'  # frechet, dtw, hausdorff, edr, st_sim

