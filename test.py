import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--ckpt',type=str)

args = parser.parse_args()

print(args)