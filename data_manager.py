class DataManager:

    @staticmethod
    def load_data(path):
        with open(path) as file:
            data = file.readlines()
        return data
