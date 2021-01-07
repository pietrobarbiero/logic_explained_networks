import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

'''
See https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
For a nice overview 

6 latent factors:   color, shape, scale, rotation and position (x and y)
'latents_sizes':    array([ 1,  3,  6, 40, 32, 32])


A primer on bases:
Assume you have a vector A = (x, y, z), where every dimension is in base (a, b, c)
Then, in order to convert each of those dimensions to decimal, we do:

D = (z * 1) + (y * c) + (x * b * c)

Or, can define bases vector B = [b*c, c, 1], and then define D = B.A

Example:
(1, 0, 1) in bases (2, 2, 2) (i.e. in binary):
D = (2^0 * 1) + (2^1 * 0) + (2^2 * 1) = 1 + 4 = 5 


'''

# Set the random state used in train/test split of data
random_state = 42



# Return the number of labels of each concept
def get_latent_sizes():
    return np.array([1, 3, 6, 40, 32, 32])


# See "primer on bases" above, to understand
def get_latent_bases():
    latents_sizes = get_latent_sizes()
    return np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                           np.array([1, ])))


# Convert a concept-based index (i.e. (color_id, shape_id, ..., pos_y_id)) into a single index
def latent_to_index(latents, latents_bases):
    return np.dot(latents, latents_bases).astype(int)



def load_dsprites(dataset_path, c_filter_fn=lambda x: True, filtered_c_ids=np.arange(6), label_fn=lambda x: x[1],
                  train_test_split_flag=False, train_size=0.85):
    '''
    :param dataset_path:  path to the .npz dsprites file
    :param c_filter_fn: function returning True/False for whether to keep the concept combination or not
    :param filtered_c_ids: np array specifying which concepts to keep in the dataset
    :param label_fn: function taking in concept values, and returning an output task label
    :param train_test_split_flag: whether to perform a train-test split or not
    :param train_size: fraction of data to use for train
    :return:
    '''

    # Load dataset
    dataset_zip = np.load(dataset_path)
    imgs = dataset_zip['imgs']

    # Compute the index conversion scheme
    latent_sizes = get_latent_sizes()
    latents_bases = get_latent_bases()

    # Get all combinations of concept values
    latent_size_listss = [list(np.arange(i)) for i in latent_sizes]
    all_combs = np.array(list(itertools.product(*latent_size_listss)))

    # Compute which concept labels to filter out
    c_ids = np.array([c_filter_fn(i) for i in all_combs])
    c_ids = np.where(c_ids == True)[0]
    c_data = all_combs[c_ids]

    # Compute the class labels from concepts
    y_data = np.array([label_fn(i) for i in c_data])

    # Get corresponding ids of these combinations in the 'img' array
    img_indices = [latent_to_index(i, latents_bases) for i in c_data]

    # Select the corresponding imgs
    x_data = imgs[img_indices]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = x_data.astype(('float32'))

    # Filter out specified concepts, with their names
    names = ['color', 'shape', 'scale', 'rotation', 'x_pos', 'y_pos']
    c_names = [names[i] for i in filtered_c_ids]
    c_data = c_data[:, filtered_c_ids]

    # If no train/test split speficied - return data as-is
    if train_test_split_flag is False:
        return x_data, y_data, c_data, c_names

    x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(x_data, y_data, c_data, train_size=train_size,
                                                                         random_state=random_state)

    x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(x_train, y_train, c_train, test_size=0.33, random_state=42)
    x_train = np.transpose(x_train, [0, 3, 1, 2])
    x_val = np.transpose(x_val, [0, 3, 1, 2])
    x_test = np.transpose(x_test, [0, 3, 1, 2])
    x_train = np.repeat(x_train, 3, axis=1)
    x_val = np.repeat(x_val, 3, axis=1)
    x_test = np.repeat(x_test, 3, axis=1)

    enc = OneHotEncoder(sparse=False)
    c_train_list, c_val_list, c_test_list = [], [], []
    for concept in range(c_train.shape[1]):
        c_train_list.append(enc.fit_transform(c_train[:, concept].reshape(-1, 1)))
        c_val_list.append(enc.fit_transform(c_val[:, concept].reshape(-1, 1)))
        c_test_list.append(enc.fit_transform(c_test[:, concept].reshape(-1, 1)))
    c_train = np.hstack(c_train_list)
    c_val = np.hstack(c_val_list)
    c_test = np.hstack(c_test_list)
    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_val = enc.fit_transform(y_val.reshape(-1, 1))
    y_test = enc.fit_transform(y_test.reshape(-1, 1))

    # c_train = c_train[:, :9]
    # c_val = c_val[:, :9]
    # c_test = c_test[:, :9]

    print('x_train shape:', x_train.shape)
    print('c_train shape:', c_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_val', x_val.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    return x_train, y_train, x_val, y_val, x_test, y_test, c_train, c_val, c_test, c_names
