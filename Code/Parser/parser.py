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
        help="Integer for latent space dimension",
    )

    parser.add_argument(
        "--batch",
        const=32,
        default=32,
        nargs="?",
        type=int,
        help="Integer for batch size",
    )

    parser.add_argument(
        "--epochs",
        const=100,
        default=100,
        nargs="?",
        type=int,
        help="Integer for training epochs",
    )

    parser.add_argument(
        "--dropout",
        const=0.2,
        default=0.2,
        nargs="?",
        type=float,
        help="Float for dropout percentage at the end of feature extractor network",
    )

    parser.add_argument(
        "--kernel_size_x",
        const=3,
        default=3,
        nargs="?",
        type=float,
        help="Integer for x-dimension of convolutional kernel",
    )

    parser.add_argument(
        "--kernel_size_y",
        const=3,
        default=3,
        nargs="?",
        type=float,
        help="Integer for y-dimension of convolutional kernel",
    )

    parser.add_argument(
        "--pool_size_x",
        const=2,
        default=2,
        nargs="?",
        type=float,
        help="Integer for x-dimension of pooling kernel",
    )

    parser.add_argument(
        "--pool_size_y",
        const=2,
        default=2,
        nargs="?",
        type=float,
        help="Integer for y-dimension of  pooling kernel",
    )

    parser.add_argument(
        "--test_size",
        const=0.2,
        default=0.2,
        nargs="?",
        type=float,
        help="Float in (0,1) representing the percentage of sample to be kept for testing",
    )

    parser.add_argument(
        "--validation_size",
        const=0.3,
        default=0.3,
        nargs="?",
        type=float,
        help="Float in (0,1) representing the percentage of sample to be kept for validation",
    )

    parser.add_argument(
        "--seed",
        const=42,
        default=42,
        nargs="?",
        type=int,
        help="Integer representing the seed used in train/test/validation splitting",
    )

    parser.add_argument(
        "--filters",
        const=(4, 8, 16, 32, 64, 128, 256),
        default=(4, 8, 16, 32, 64, 128, 256),
        nargs="?",
        type=tuple,
        help="Filters to apply in CNN",
    )

    # Parse arguments
    args = parser.parse_args()

    return args
