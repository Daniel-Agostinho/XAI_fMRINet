# Built-in imports
from itertools import product

# My imports
from src.interpretability_statistics.statistical_analysis import explainable_maps_statistics


def main():
    atlases = ["BN", "HCP"]

    approach = [3, 4]
    args = list(product(atlases, approach))

    # Statistical analysis
    for arg in args:
        explainable_maps_statistics(*arg)


if __name__ == '__main__':
    main()
