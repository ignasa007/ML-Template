# import model classes


map = {
    # model name: model class
}


def dataclass_map(dataset_name):
    dataset_name = dataset_name.lower()
    return map[dataset_name]