class Results:

    def __init__(self):
        self.results = None

    def update_results(self):
        raise NotImplementedError

    def compute_metrics(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError