import argparse
from polynomial_fitting import train


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=1, type=int, help="Part of code to run")
    parser.add_argument("--method", default="gd", help = "Type of solver")  
    parser.add_argument("--lr", default=0.01, type=float, help = "Learning Rate for gradient descent")
    parser.add_argument("--epochs", default=100, type=int, help = "batch size")
    parser.add_argument("--batch_size", default=10, type=int, help = "Batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "Regularization constant")
    parser.add_argument("--polynomial", default=10, type=int, help = "Degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="data/gaussian.csv", type=str, help = "Read content from the file")
    parser.add_argument("--split", default=0.2, type=float, help = "Split for train/test set")
    return parser.parse_args()
    

if __name__ == '__main__':
    args = setup()
    print(args)
    if args.part == 1:
        train.fit_polynomial(args)
    