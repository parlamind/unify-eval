import argparse
from dataclasses import dataclass


@dataclass
class ParsedArgs:
    """
    data class storing training script arguments
    """
    initial: int
    iterations: int
    minibatch_size: int
    language: str


def parse_args():
    """
    parses script arguments:
    -initial iteration index
    -number of training iterations
    -size of minibatch during training,
    -initial iteration index
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial",
                        "-i",
                        help="initial iteration index. if 0, start new model from scratch",
                        type=int,
                        default=0)
    parser.add_argument("--iterations",
                        "-n",
                        help="number of training iterations",
                        type=int,
                        default=100)
    parser.add_argument("--minibatch_size",
                        "-m",
                        help="size of minibatch during training",
                        type=int,
                        default=50)
    parser.add_argument("--language",
                        "-l",
                        help="initial iteration index. if 0, start new model from scratch",
                        choices=["de", "en"],
                        type=str,
                        default="de")
    args = parser.parse_args()

    return ParsedArgs(initial=args.initial,
                      iterations=args.iterations,
                      minibatch_size=args.minibatch_size,
                      language=args.language)
