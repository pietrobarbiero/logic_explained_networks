from dSprites_data_loader import load_dsprites


def concept_filters(schema):
    '''
    Specify which concept values to filter out
    :param schema: Name of the schema
    :return: filtered out values for all the concepts, except the color concept
    '''

    if schema is "full":
        shape_range = list(range(3))
        scale_range = list(range(6))
        rot_range = [i for i in range(0, 40, 1)]
        x_pos_range = [i for i in range(0, 32, 1)]
        y_pos_range = [i for i in range(0, 32, 1)]

    elif schema is "small_skip":
        shape_range = list(range(3))
        scale_range = list(range(6))
        rot_range = [i for i in range(0, 40, 5)]
        x_pos_range = [i for i in range(0, 32, 2)]
        y_pos_range = [i for i in range(0, 32, 2)]

    elif schema is "big_skip":
        shape_range = list(range(3))
        scale_range = list(range(6))
        rot_range = [i for i in range(0, 40, 10)]
        x_pos_range = [i for i in range(0, 32, 10)]
        y_pos_range = [i for i in range(0, 32, 10)]

    else:
        raise NotImplemented()

    return shape_range, scale_range, rot_range, x_pos_range, y_pos_range


# ===========================================================================
#                   Task DEFINITIONS
# ===========================================================================

'''
Define the label functions, which take concept values as input, and return task labels as output
All of these functions take 
'''

def get_shape_label(concepts):
    '''
    The 'shape' task, in which the task label is simply the shape concept label
    '''
    return concepts[1]



def get_shape_scale(shape_range, scale_range):
    '''
    Assign a unique class to every combination in shape_range and scale_range
    :param shape_range:
    :param scale_range:
    :return:
    '''
    label_map = {}
    cnt = 0
    n_scales = len(scale_range)

    for sh in shape_range:
        for sc in scale_range:
            key = sh * n_scales + sc
            label_map[key] = cnt
            cnt += 1

    def label_fn(concepts):
        key = concepts[1] * n_scales + concepts[2]
        return label_map[key]

    return label_fn




def get_data(args):

    # Specify the path to the dsprites data file
    path = args.dsprites_path

    # Load dataset specified by schema
    dataset_schema = 'small_skip'
    # dataset_schema = 'big_skip'

    # Get filtered concept values
    shape_range, scale_range, rot_range, x_pos_range, y_pos_range = concept_filters(dataset_schema)

    # Define function for filtering out specified concept values only
    def c_filter_fn(concepts):
        in_shape_range = (concepts[1] in shape_range)
        in_scale_range = (concepts[2] in scale_range)
        in_rot_range = (concepts[3] in rot_range)
        in_x_range = (concepts[4] in x_pos_range)
        in_y_range = (concepts[5] in y_pos_range)

        return (in_shape_range and in_scale_range and in_rot_range and in_x_range and in_y_range)


    # Get label function, assigning task labels for concept values
    task_name = args.task_name

    if task_name == 'shape':
        label_fn = get_shape_label
    elif task_name=='shape_scale':
        label_fn = get_shape_scale(shape_range, scale_range)
    else:
        raise NotImplementedError()

    # Load dataset
    x_train, y_train, x_val, y_val, x_test, y_test, c_train, c_val, c_test, c_names = load_dsprites(path,
                                                                               c_filter_fn=c_filter_fn,
                                                                               label_fn=label_fn,
                                                                               train_test_split_flag=True)


    return x_train, y_train, x_val, y_val, x_test, y_test, c_train, c_val, c_test, c_names

