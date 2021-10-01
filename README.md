# Scale-Preserving Automatic Concept Extraction (SPACE)

The SPACE algorithm is a technique for automatically extracting concept-based explanations from deep learning models for image classification. A central algorithmic requirement that led the design process of the SPACE algorithm was the preservation of scale of all features while extracting concepts and evaluating their importance. SPACE is able to identify model as well as dataset problems occurring in image classification tasks and can therefore help to increase the reliability of machine learning models utilized for such purposes.

## Installation

As an example, into an environment with Python 3.7.11 you can install all needed packages with:

`pip install -r requirements.txt`

## Usage

See **SPACE_example_notebook.ipynb** for an example notebook showing step by step how to use the SPACE code. Alternatively, this notebook can also be directly launched in [Google Colab](https://colab.research.google.com/github/lkreiskoether/SPACE/blob/master/SPACE_example_notebook.ipynb).
