import numpy as np


def save_grids_strides(width, height):
    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    hsizes = [height // stride for stride in strides]
    wsizes = [width // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)

    # save grids as grids.npy
    np.save('grids.npy', grids)

    # save expanded_strides as expanded_strides.npy
    np.save('expanded_strides.npy', expanded_strides)

    print('grids.shape', grids.shape)
    print('expanded_strides.shape', expanded_strides.shape)


if __name__ == '__main__':
    from sys import argv

    if len(argv) == 3:
        save_grids_strides(int(argv[1]), int(argv[2]))
        print('Done!')
    else:
        print('Usage: python build_npy_files.py <width> <height>')
