import argparse
from argparse import ArgumentParser


def parse():

    """Parse command line arguments"""

    # Initiate argparser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--latent",
        const=48,
        default=48,
        nargs="?",
        type=int,
        help="Integer for latent space dimension"
    )

    parser.add_argument(
        "--batch",
        const=32,
        default=32,
        nargs="?",
        type=int,
        help="Integer for batch size"
    )

    parser.add_argument(
        "--epochs",
        const=100,
        default=100,
        nargs="?",
        type=int,
        help="Integer for training epochs"
    )

    parser.add_argument(
        "--dropout",
        const=0.2,
        default=0.2,
        nargs="?",
        type=float,
        help="Float for dropout percentage at the end of feature extractor network"
    )
   
    # Parse arguments
    args = parser.parse_args()

    return args