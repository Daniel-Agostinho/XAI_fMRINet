from .networks.cnn import FMRINet


def initialize_model(network_name, parameters):

    if network_name == "fMRINet":
        return FMRINet(**parameters)

    raise ValueError("Invalid Network")


if __name__ == '__main__':
    pass
