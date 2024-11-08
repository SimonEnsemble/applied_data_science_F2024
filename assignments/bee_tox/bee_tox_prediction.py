import marimo

__generated_with = "0.9.16"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    import numpy as np

    import pandas as pd

    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import ConfusionMatrixDisplay, f1_score, balanced_accuracy_score

    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, Draw

    import matplotlib.pyplot as plt
    import seaborn as sns
    from aquarel import load_theme

    theme = load_theme("arctic_light").set_overrides({
        'lines.linewidth': 3,
        'font.size': 18
    })
    theme.apply()

    my_palette = sns.color_palette("pastel")
    return (
        Chem,
        ConfusionMatrixDisplay,
        DecisionTreeClassifier,
        Draw,
        MACCSkeys,
        balanced_accuracy_score,
        cross_val_score,
        f1_score,
        load_theme,
        mo,
        my_palette,
        np,
        pd,
        plt,
        sns,
        theme,
        train_test_split,
        tree,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # ::noto:honeybee:: classifying the toxicity of pesticides to bees

        !!! objective
            our goal is to train and test a pruned decision tree classifier to predict the toxicity of pesticides to honey bees. the input to the classifier is a feature vector describing the molecular structure, and the output is a binary classification "toxic" or "non-toxic". 

            ::icon-park:data:: to enable machine-learning of bee toxicity, we rely on the ApisTox data set [here](https://github.com/j-adamczyk/ApisTox_dataset/blob/master/outputs/dataset_final.csv). this provides examples of molecules and their toxicity to honey bees.

            ::fxemoji:lightbulb:: see the introduction of Cory's paper on predicting the toxicity of pesticides to honey bees [here](https://pubs.aip.org/aip/jcp/article/157/3/034102/2841476) for a motivation.

        ## read in, explore data

        ::noto:honeybee:: use `pandas` to read in the ApisTox data set as a `DataFrame`. replace the "0"'s and "1"'s in the label column with more informative labels "non-toxic" and "toxic".
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""::noto:honeybee:: draw a bar plot showing the class distribution (i.e. the number of molecules in the data set that are toxic to bees and the number that are nontoxic. so two bars.)""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""::noto:honeybee:: draw draw a bar plot visualizing the number of molecules in the data set that fall in the herbicide, insecticide, and fungicide catgories (so, three bars).""")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""::noto:honeybee:: how many molecules are in the data set? are the molecules unique, judged by the SMILES strings?""")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""::noto:honeybee:: is the molecule with the name "Nicotine" toxic to bees? draw its molecular structure from the SMILES string in the data frame using `rdkit` (see [here](https://www.rdkit.org/docs/GettingStartedInPython.html)).""")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## create feature matrix and target vector

        !!! molecular "molecular representations"
            molecules are not vectors, but decision trees take vectors as inputs. we will represent each molecular structure with a vector indicating the presence or absence of a list of molecular substructures---specifically, MACCS keys. this means we represent each molecule as a 167-dimensional bit vector. see [here](https://chem.libretexts.org/Courses/Intercollegiate_Courses/Cheminformatics/06%3A_Molecular_Similarity/6.01%3A_Molecular_Descriptors) to learn more about molecular descriptors and MACCS keys.

        ::noto:honeybee:: to prepare for machine learning, use all data to create:

        * the target vector `y`, a list of the class labels (toxic or non-toxic) on the molecules. pull these labels from the data frame.
        * the feature matrix `X`, a matrix of features of the molecules. the feature vectors of the molecules---the MACCS fingerprints---should lie in the rows. so the number of rows is the number of molecules, and the number of columns is the number of MACCS keys (features). use `rdkit` to search for the MACCS keys in the molecular structures to create the feature vectors of them. (see [here](https://www.rdkit.org/docs/GettingStartedInPython.html#maccs-keys).)

        so entry $i$ of the target vector is the label on the molecule represented by row $i$ in the feature matrix.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    # pre-allocate the feature matrix

    # loop thru molecules

        # get SMILES string for this molecule

        # convert the SMILES to an RDkit molecule

        # compute MACCS feature vector,
        #  assign as row m in feature matrix

    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ::noto:honeybee:: analyze/visualize the feature matrix in three ways: 

        * draw a heatmap to directly visualize the feature matrix using matplotlib's `imshow` function. label the x- and y-axis as corresponding to either "molecule" or "feature".
        * draw a histogram of the number of bits activated in the MACCS feature vector among the molecules
        * draw a histogram of the number of molecules that activate a bit in the the MACCS feature vector among the features.

        !!! hint
            I say a bit in the feature vector for a molecule is "activated" if the corresponding MACCS substructure is present in the molecule. to compute these for the two histograms, sum along the rows or columns of the feature matrix.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## train/test split

        ::noto:honeybee:: randomly partition the target vector and feature matrix into a 67%/33% train/test set, stratified by class. so you should have variables `X_train, X_test` giving the feature matrices for the train and test set and `y_train, y_test` giving the corresponding target vectors. also split the names of the molecules to check which molecules are in train set and which are in the test set.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""::noto:honeybee:: is nicotine in the train set or test set?""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## ::ph:tree-duotone:: the cost-complexity pruning path of the trained decision tree

        ::noto:honeybee:: follow  [this scikit-learn guide](https://scikit-learn.org/1.5/auto_examples/tree/plot_cost_complexity_pruning.html) to:

        1. compute the cost-complexity pruning path of a decision tree classifier trained on the training data.
        2. plot the sum of Gini impurities of the leaves in the pruned tree against the complexity parameter $\alpha$.
        3. plot (a) the number of nodes and (b) the depth of the trained decision tree against the complexity parameter $\alpha$ along the pruning path.


        !!! note
            pause and interpret these plots. the larger the complexity parameter $\alpha$, the less complex (i.e. fewer nodes and less depth) the tree becomes, yet more impure the data in its leaves become. so, each $\alpha$ strikes some tradeoff between fit on training data and tree complexity.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## ::ph:tree-duotone:: cross-validation to determine the optimal complexity parameter for pruning

        ::noto:honeybee:: use $K=4$-folds cross-validation on the training data to determine the optimal complexity parameter $\\alpha$ based on the mean balanced accuracy. plot the mean balanced accuracy over the four test folds against the complexity parameter $\\alpha$. what is the optimal value for $\\alpha$ based on your cross-validation routine? see the [scikitlearn guide](https://scikit-learn.org/dev/modules/cross_validation.html#cross-validation) on cross validation.

        !!! hint
            an intermediate complexity parameter should provide the largest balanced accuracy. 

            the candidate set of $\\alpha$'s to cross-validate over are given by the cost-complexity pruning path above.

            the mean balanced accuracy is not the default score in `cross_val_score`.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## ::ph:tree-duotone:: train the decision tree of the optimal complexity for testing

        ::noto:honeybee:: finally, train a decision tree classifier on _all_ of the training data using the _optimal_ complexity parameter $\alpha$ determined from the cross-validation routine. this is the decision tree we will use for testing.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## ::ph:tree-duotone:: test the trained decision tree

        ::noto:honeybee:: use the trained and optimally-pruned decision tree (both done using the training data only) to predict labels on the held-out test set of molecules. compare the predicted labels on the test set of molecules to the true labels by:

        1. drawing a confusion matrix.
        2. computing the balanced accuracy over the test set.

        since the test data was held out from development of the decision tree, both reflect the performance we can expect from the tree on new pesticide molecules (assuming no [distribution shift](https://docs.google.com/presentation/d/1tuCIbk9Pye-RK1xqiiZXPzT8lIgDUL6CqBkFSYZXkbY/edit#slide=id.p)).
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## ::ph:tree-duotone:: deployment tree

        we now have an estimate of the performance of the decision tree for classifying the toxicity of pesticides to bees. the test/train split was necessary for this. 

        for the decision tree we'll deploy in the wild, to make predictions on molecules with unknown toxicity to bees and make decisions such as "go ahead and use that molecule as a pesticide", we shouldn't let any of our data go to waste. generally, machine learning models improve with more data.

        ::noto:honeybee:: train a _deployment_ decision tree on _all_ of the data using the optimal cost-complexity parameter determined from cross-validation on the train set. this is the tree we'd use in the wild, for new pesticides, if we actually wanted to make a prediction and make decisions.

        !!! note
            actually, we'd probably want to do cross-validation _again_ on _all_ of the data, to determine the optimal cost complexity parameter for training on _all_ of the data, _then_ train on all data using _that_ (perhaps, different) cost-complexity parameter. but, I'll save you the trouble...
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## ::ph:tree-duotone:: interpreting the tree

        ::noto:honeybee:: use `scikitlearn`'s `plot_tree` function to visualize the structure of the tree.

        !!! note
            not necessary, but if you were curious like me, see [here](https://stackoverflow.com/questions/59447378/sklearn-plot-tree-plot-is-too-small) for how to save the tree as a PDF so you can open it in an external program and zoom in to see the details of each node.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        suppose we were considering to use the molecule cannabidiol (CBD) as a fungicide. we wish to predict if CBD is toxic to bees or not _and_ explain the prediction by the decision tree. (hooray, decision trees are interpretable!)


        ::noto:honeybee:: use `rdkit` to obtain the MACCS fingerprint for CBD. it SMILES string is CCCCCC1=CC(=C(C(=C1)O)[C@@H]2C=C(CC[C@H]2C(=C)C)C)O.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ::noto:honeybee:: use your deployed decision tree to predict if CBD is toxic to bees or not.

        !!! hint
            you'll probably need a `.reshape`, since the `DecisionTreeClassifier.predict` method is designed to make predictions on multiple inputs at once, not just one.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ::noto:honeybee:: follow [scikit-learn's guide](https://scikit-learn.org/1.5/auto_examples/tree/plot_unveil_tree_structure.html#decision-path) to print the decision path used to make this classification of CBD---i.e., how the CBD fingerprint input to the root node of the tree percolates down the tree to a leaf node where it gets classified as toxic or not. each decision at a node should be printed in an interpretable way e.g. "the molecular structure of CBD exhibits/lacks the pattern [#16]~*(~*)~*".

        !!! hint
            the corresponding SMARTS patterns for the MACCS keys are obtained via `MACCSkeys.smartsPatts`.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
