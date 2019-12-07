import os
import os.path
import argparse
from util import *
from scipy.io import savemat

if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='Postprocessing .mat voxel file.')
    cmd_parser.add_argument('-t', '--threshold', metavar='threshold', type=float, default=0.1, help='voxels with confidence lower than the threshold are not displayed')
    cmd_parser.add_argument('-i', '--index', metavar='index', type=int, default=1, help='the index of objects in the inputfile that should be rendered (one based)')
    cmd_parser.add_argument('filename', metavar='filename', type=str, help='name of .mat file to be visualized')
    cmd_parser.add_argument('-df', '--downsample-factor', metavar='factor', type=int, default=1, help='downsample objects via a max pooling of step STEPSIZE for efficiency. Currently supporting STEPSIZE 1, 2, and 4.')
    cmd_parser.add_argument('-dm', '--downsample-method', metavar='downsample_method', type=str, default='max', help='downsample method, where mean stands for average pooling and max for max pooling')
    cmd_parser.add_argument('-mc', '--max-component', metavar='max_component', type=int, default=3, help='whether to keep only the maximal connected component, where voxels of distance no larger than `DISTANCE` are considered connected. Set to 0 to disable this function.')
    
    args = cmd_parser.parse_args()
    filename = args.filename
    matname = 'voxels'
    threshold = args.threshold
    ind = args.index - 1  # matlab uses 1 base index
    downsample_factor = args.downsample_factor
    downsample_method = args.downsample_method
    connect = args.max_component

    assert downsample_method in ('max', 'mean')

    # read file
    print('==> Reading input voxel file: ' + filename)
    voxels_raw = read_tensor(filename, matname)
    print('Done')

    voxels = voxels_raw[ind]

    # keep only max connected component
    if connect > 0:
        print('==> Looking for max connected component')
        voxels_keep = (voxels >= threshold)
        voxels_keep = max_connected(voxels_keep, connect)
        voxels[np.logical_not(voxels_keep)] = 0
        print('Done')

    # downsample if needed
    if downsample_factor > 1:
        print('==> Performing downsample: factor: ' + str(downsample_factor) + ' method: ' + downsample_method)
        voxels = downsample(voxels, downsample_factor, method=downsample_method)
        print('Done')

    # binarize
    voxels[voxels >= threshold] = 1
    voxels[voxels < 1] = 0

    # save new mat file
    dot_idx = filename.rfind('.')
    if dot_idx >= 0:
        out_filepath = filename[:dot_idx] + '_postprocessed' + filename[dot_idx:]
    else:
        out_filepath = filename + '_postprocessed'
    print('==> Writing %s' % out_filepath)
    savemat(out_filepath, {matname: voxels})
    print('Done')
