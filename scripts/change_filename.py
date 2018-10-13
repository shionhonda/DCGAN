import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='../images/dressed', help='Path to the images')
    args = parser.parse_args()

    path = args.path
    files = glob.glob(path + '/*.jpg')
    for i, f in enumerate(files):
        os.rename(f, os.path.join(path, '{0:04d}'.format(i)+'.jpg'))

if __name__ == '__main__':
    main()
