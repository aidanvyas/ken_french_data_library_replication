"""
A special thanks to Qingyi (Freda) Song Drechsler whose code on WRDS serves as the base of this implementation.
"""

from process_data import process_data
from replicate_fama_french import replicate_fama_french

def main():
    """
    Process the data and replicate the Fama-French factors.
    """

    process_data()
    replicate_fama_french()


if __name__ == "__main__":
    main()
