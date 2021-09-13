import argparse


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=1, type=int, help="Part of code to run")
    parser.add_argument("--method", default="pinv", help = "Type of solver")  
    parser.add_argument("--batch_size", default=5, type=int, help = "Batch size")
    parser.add_argument("--lambda", default=0, type=float, help = "Regularization constant")
    parser.add_argument("--polynomial", default=10, type=float, help = "Degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()
    

if __name__ == '__main__':
    args = setup()
    print(args)
    