from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """
    H, W, C = image.shape
    nums = np.random.randint(H * W, size=num_clusters)
    centroids_init = image.reshape(-1, C)[nums]
    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """
    num_clusters = len(centroids)
    H, W, C = image.shape
    image = image.reshape(-1, C)
    dist = np.empty([num_clusters, H * W])
    converged = False
    
    for it in range(max_iter):
        # Do E-step
        for j in range(num_clusters):
            dist[j] = np.sum((image - centroids[j]) ** 2, axis=1)
        clustering = np.argmin(dist, axis=0).reshape(-1, 1)
        # Do M-step
        new_centroids = np.empty([num_clusters, C])
        for j in range(num_clusters):
            cluster_j = (clustering == j)
            new_centroids[j] = np.sum(cluster_j * image, axis=0) / np.sum(cluster_j)
        # print loss
        if (it + 1) % print_every == 0:
            loss = (image - new_centroids[clustering.squeeze()]) ** 2
            loss = np.sum(loss)
            print(f'loss: {loss:.2f}')
        # check convergence
        if np.array_equal(centroids, new_centroids):
            converged = True
            break
        centroids = new_centroids
        
    if converged:
         print(f'Converged after {it + 1} iterations')
    else:
        print(f"Still didn't converged after {it + 1} iteration")
    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    new_image : nparray
        Updated image
    """
    num_clusters = len(centroids)
    H, W, C = image.shape
    image = image.reshape(-1, C)
    dist = np.empty([num_clusters, H * W])
    for j in range(num_clusters):
        dist[j] = np.sum((image - centroids[j]) ** 2, axis=1)
    clustering = np.argmin(dist, axis=0)
    new_image = centroids[clustering].reshape(H, W, C)
    return new_image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small)) / 255
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.axis('off')
    plt.imshow(image)
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large)) / 255
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.axis('off')
    plt.imshow(image)
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.axis('off')
    plt.imshow(image_clustered)
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
