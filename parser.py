"""Parsing the model parameters."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Twitch Brasilians dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node identifiers.
    """
    parser = argparse.ArgumentParser(description="Run DANMF.")

    # nargs="?"表示参数可以设置零个或一个
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="../input/ptbr_edges.csv",
	                help="Edge list csv.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="../output/ptbr_danmf.csv",
	                help="Target embedding csv.")

    parser.add_argument("--membership-path",
                        nargs="?",
                        default="../output/ptbr_membership.json",
	                help="Cluster membership json.")

    parser.add_argument("--iterations",
                        type=int,
                        default=100,
	                help="Number of training iterations. Default is 100.")

    parser.add_argument("--pre-iterations",
                        type=int,
                        default=1000,
	                help="Number of layerwsie pre-training iterations. Default is 1000.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--lamb",
                        type=float,
                        default=0,
	                help="Regularization parameter. Default is 0.01.")

    # nargs="+"表示参数可以设置一个或多个
    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 128 64 32.")
    # dest 给参数起了个别名
    parser.add_argument("--calculate-loss",
                        dest="calculate_loss",
                        action="store_true")

    parser.add_argument("--not-calculate-loss",
                        dest="calculate_loss",
                        action="store_false")

    parser.set_defaults(calculate_loss=False)
    # 512, 258, 128, 64, 32 # 258, 64
    parser.set_defaults(layers=[128, 86, 64]) # 64 , 50 , 32, 16 # 64 , 32 , 50 , 20

    return parser.parse_args()
