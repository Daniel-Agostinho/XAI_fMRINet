# Built-in imports
import os
from itertools import product

# My imports
from src.interpretability.explain_models import model_explanation_approach


def main():
    atlases = ["BN", "HCP"]

    approach = [6]
    args = list(product(atlases, approach))

    for arg in args:
        model_explanation_approach(*arg)


if __name__ == '__main__':
    main()
