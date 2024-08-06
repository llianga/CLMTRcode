import argparse

parser = argparse.ArgumentParser(description="Cons_St2vec")
parser.add_argument('--config_file', default='configs/semantic_tdrive_batch256.yaml', metavar="FILE", help="path to config file", type=str)

# parser.add_argument("-feature_size", type=int, default=64, help="the weight of contrastive time loss")

# parser.add_argument("-embedding_size", type=int, default=64, help="the weight of contrastive time loss")

# parser.add_argument("-time2vec_size", type=int, default=64, help="the weight of contrastive time loss")

# parser.add_argument("-hidden_size", type=int, default=128, help="the weight of contrastive time loss")

# parser.add_argument("-num_layers", type=int, default=1, help="the weight of contrastive time loss")

# parser.add_argument("-dropout_rate", type=int, default=0, help="the weight of contrastive time loss")

# parser.add_argument("-concat", type=bool, default=False, help="the weight of contrastive time loss")

# parser.add_argument("-de_embedding_size", type=int, default=64, help="the weight of contrastive time loss")

# parser.add_argument("-n_direction", type=int, default=2, help="the weight of contrastive time loss")

# parser.add_argument("-vocab_size", type=int, default=51182, help="the weight of contrastive time loss")

# parser.add_argument("-timevocab_size", type=int, default=52605, help="the weight of contrastive time loss")

# parser.add_argument("-learning_rate", type=float, default=0.001, help="the weight of contrastive time loss")

# parser.add_argument("-train_epochs", type=int, default=150, help="the weight of contrastive time loss")

# parser.add_argument("-batch_size", type=int, default=10, help="the weight of contrastive time loss")

# parser.add_argument("-print_freq", type=int, default=50, help="the weight of contrastive time loss")

# parser.add_argument("-save_freq", type=int, default=2, help="the weight of contrastive time loss")

#parser.add_argument("-logdir", default="/home3/xianghao/code_laq/Cons_St2vec/logs", help="the weight of contrastive time loss")
# parser.add_argument("-logdir", default="/home/xianghao/code_laq/Cons_St2vec/logs", help="the weight of contrastive time loss")
# parser.add_argument("-data_path", default="/home2/xianghao/code_laq/Cons_St2vec/data",
#     help="Path to training, validating and testing data")

# parser.add_argument("-checkpoint", default="/home2/xianghao/code_laq/Cons_St2vec/data/checkpoint.pt",
#     help="The saved checkpoint")

# parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")

# parser.add_argument("-pretrained_embedding", default=None,
#     help="Path to the pretrained word (cell) embedding")

# parser.add_argument("-num_layers", type=int, default=1,
#     help="Number of layers in the RNN cell")

# parser.add_argument("-bidirectional", type=bool, default=False,
#     help="True if use bidirectional rnn in encoder")

# parser.add_argument("-hidden_size", type=int, default=256,
#     help="The hidden state size in the RNN cell")

# parser.add_argument("-embedding_size", type=int, default=256,
#     help="The word (cell) embedding size")

# parser.add_argument("-dropout", type=float, default=0.2,
#     help="The dropout probability")

# parser.add_argument("-max_grad_norm", type=float, default=5.0,
#     help="The maximum gradient norm")

# parser.add_argument("-learning_rate", type=float, default=0.001)

# parser.add_argument("-batch", type=int, default=2,
#     help="The batch size")

# parser.add_argument("-generator_batch", type=int, default=32,
#     help="""The maximum number of words to generate each time.
#     The higher value, the more memory requires.""")

# parser.add_argument("-t2vec_batch", type=int, default=256,
#     help="""The maximum number of trajs we encode each time in t2vec""")

# parser.add_argument("-start_iteration", type=int, default=0)

# parser.add_argument("-epochs", type=int, default=15,
#     help="The number of training epochs")

# parser.add_argument("-print_freq", type=int, default=50,
#     help="Print frequency")

# parser.add_argument("-save_freq", type=int, default=1000,
#     help="Save frequency")

# parser.add_argument("-cuda", type=bool, default=True,
#     help="True if we use GPU to train the model")

# parser.add_argument("-use_discriminative", action="store_true",
#     help="Use the discriminative loss if the argument is given")

# parser.add_argument("-discriminative_w", type=float, default=0.1,
#     help="discriminative loss weight")

# parser.add_argument("-criterion_name", default="NLL",
#     help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")

# parser.add_argument("-knearestvocabs", default=None,
#     help="""The file of k nearest cells and distances used in KLDIVLoss,
#     produced by preprocessing, necessary if KLDIVLoss is used""")

# parser.add_argument("-dist_decay_speed", type=float, default=0.8,
#     help="""How fast the distance decays in dist2weight, a small value will
#     give high weights for cells far away""")

# parser.add_argument("-max_num_line", type=int, default=20000000)

# parser.add_argument("-max_length", default=20000,
#     help="The maximum length of the target sequence")

# parser.add_argument("-mode", type=int, default=0,
#     help="Running mode (0: train, 1:evaluate, 2:t2vec)")

# parser.add_argument("-vocab_size", type=int, default=0,
#     help="Vocabulary Size")

# parser.add_argument("-bucketsize", default=[(200, 400), (300, 400),(400,500), (500, 600),(600,700), (600, 800), (700, 900),(900,1000),(1000,1100),(1100,1200)],
#     help="Bucket size for training")

def get_args():
    args, unparsed_args = parser.parse_known_args()
    # args = parser.parse_args()
    return args
