'''
__init__ file for data classes.
imports all the data classes and creates a function to map the 
    dataset name to the data class.
'''

# import data classes


map = {
    # data name: data class
}


def dataclass_map(dataset_name):

    '''
    Function to map dataset name to data class.

    Args:
        dataset_name (str): name of the dataset used for the experiment
    
    Return:
        data_class (BaseDataset): a data class if dataset_name is 
            recognized, else None
    '''
    
    data_class = map.get(dataset_name.lower(), None)
    
    return data_class