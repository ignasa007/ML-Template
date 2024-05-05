'''
__init__ file for model classes.
imports all the model classes and creates a function to map the 
    model name to the model class.
'''

# import model classes


map = {
    # model name: model class
}


def modelclass_map(model_name):

    '''
    Function to map model name to model class.

    Args:
        model_name (str): name of the model used for the experiment
    
    Return:
        model_class (BaseModel): a model class if model_name is 
            recognized, else None
    '''
    
    model_class = map.get(model_name.lower(), None)
    
    return model_class