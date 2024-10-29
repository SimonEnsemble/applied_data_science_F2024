import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    import seaborn as sns
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
    import matplotlib.pyplot as plt
    return (
        ConfusionMatrixDisplay,
        StratifiedKFold,
        accuracy_score,
        mo,
        np,
        pd,
        plt,
        sns,
        train_test_split,
        tree,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # ::icon-park-outline:data:: the glass data set

        !!! objective
            we wish to build a classifier that predicts whether a sample of glass from a [vehicle or building] window was [float-processed](https://en.wikipedia.org/wiki/Float_glass) or not float-processed based on measurements of its refractive index and composition. this binary classifier can then be used for criminal investigations where shattered glass is left at the scene of a crime, which is useful evidence only if we can identify the glass.

        ::noto:window:: from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/42/glass+identification), download the glass identification data set. 

        * read in measurements and class labels from `glass.data` as a pandas data frame. each row represents a glass sample.
        * give the columns (giving features and labels) appropriate names using the `names` argument of `pandas.read_csv`. info about the columns is contained in `glass.names`.
        * drop all rows pertaining to samples of containers, tableware, and headlamps. i.e. keep rows pertaining to only vehicle or building _window_ glass.
        * create a new column, named "class", that assigns a binary label, "float" or "not float", to each sample of glass (based on the three distinct categories of glass samples present at this point.)
        * remove the category and sample ID columns to avoid distraction. we only need the features of the glass samples and the binary class labels to train and test our machine learning classifier.
        * sort the rows of the data frame according to the refractive index of the sample---from low to high.

        your final data frame (163 rows) should look like: 

        | | RI | Na | Mg | Al | Si | K | Ca | Ba | Fe | class |
        | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
        | 56 | 1.51215 | 12.99 | ... | ... | ... | ... | ... | ... | ... | float |
        | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
        | 107 | ... | ... | ... | ... | ... | ... | ... | ... | ... | not float |
        """
    )
    return


@app.cell
def __():
    # read raw data

    # drop rows not pertaining to window glass samples

    # make "class" column giving binary labels "not float" or "float"
    #  (lumping irrelevant distinction between building vs. vehicle glass together.)

    # sort rows according to the refractive index

    # drop the ID and category column
    #    the only columns we need are for the features and the class

    # display the data frame

    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""::noto:window:: are there any rows with missing data? if so, drop those rows.""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""🪟 for convenience, assign a variable `features` as the list of names of columns in the data frame pertaining to the features (omitting the column giving the class labels). use `features` later to grab only the columns pertaining to the features.""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # data exploration ::ep:ship::

        ::noto:window:: how many samples of glass belong to each class (float vs. not float)? draw a bar plot to visualize how the samples are distributed among the two classes. (_hint_: use `pandas.Series.value_counts()` and `seaborn.barplot` or `seaborn.countplot`.)
        """
    )
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
        ::noto:window:: use `seaborn.pairplot` to visualize how the features of the glass samples (a) are correlated with each other in a pairwise manner and (b) distributed---both, broken down according to the class of the glass. i.e, draw a 9x9 plot matrix showing (i) pairwise scatter plots on the off-diagonals, with points colored and shaped according to the class, and (ii) distributions of the features on the diagonal, grouped by class (so, two histograms per feature). avoid information overload and redundancy by making the `pairplot` a lower triangular plot matrix via `corner=True`.

        ::noto:light-bulb:: the success of a machine learning classifier is predicated upon our ability to distinguish/separate/discriminate the two classes of glass according to the features we measured. spot any patterns hinting at the two classes of the glasses being (to a degree) separated in feature space? does any particular feature look particularly predictive of the class?
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
        # partition the data into train/test split

        ::noto:window:: use `scikitlearn.train_test_split` to randomly partition the data into a train set and test set in a class-stratified manner. allocate 1/3 of the data to the test set, and 2/3 of the data to the train set. stratification ensures that the class distribution is the same in the train and test set. please use `random_state=97330` so we/you all get the same split. 

        ::noto:light-bulb:: the result of the split should be two data frames containing data on distinct glass samples---one data frame for training the machine learning algorithm, the other for assessing the performance of the trained machine learning algorithm on glass samples it hasn't "seen" before.
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
        ::noto:window:: to check your test/train split, display: 

        1. the number of glass samples in the train and test set, separately.
        2. the fraction of glass samples that belong to each class in the train and test set, separately.
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
        # implementing a decision stump from scratch ::tdesign:tree-round-dot::

        !!! objective
            to intimately understand the innerworkings of widely used decision tree classifiers, we will code up a [_decision stump_](https://en.wikipedia.org/wiki/Decision_stump). our code will mimic how `scikit-learn` implements machine learning algorithms. specifically, we will write a `class DecisionStump` with a `.fit(X_train, y_train)` and `.predict(X)` method attached to it. 

            we will then "fit" a decision stump classifier on the training portion of the glass data set, then assess its predictive performance on the test portion of the data set. "fitting" constitutes finding the optimal parameters of the decision stump: (1) the feature for the split; (2) the threshold for the split; (3) the label for data falling in the left leaf node; and (4) the label for the data falling in the right leaf node. the "optimal" split is defined as the one giving the minimal [weighted Gini impurity](https://scikit-learn.org/dev/modules/tree.html#mathematical-formulation).

            a concrete example of a (suboptimal) decision stump for the glass classification task is below. like all decision stumps, it has one internal node and two leaf nodes (where a labels is assigned).
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    diagram = '''
    flowchart TD
        A[feature vector of glass sample] --> B[is Ca &le; 5 wt%?]
        B -->|Yes| C[float-processed]
        B -->|No| D[not float-processed]

    style A fill:white,stroke-width:0px;
    style B fill:white;
    style C fill:lightgreen;
    style D fill:lightgreen;
    '''
    mo.mermaid(diagram)
    return (diagram,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ::noto:window:: implement a `class DecisionStump`: 

        * with a `.fit(X_train, y_train)` method that, given the `DataFrame`, `X_train`, containing features (only---not the class labels) of glass samples and `Series`, `y_train`, giving the associated class labels on the glass samples, determines the optimal parameters of the decision stump and stores them as attributes. to do so, you need an outer loop over the features and an inner loop over splitting thresholds. choose the (feature, threshold) combination that gives the minimal weighted Gini impurity.
        * after each chunk of code inside the outer loop runs, print the results i.e. the best threshold and minimal weighted Gini impurity for that feature.
        * after fit is called, you should have access to attributes `feature`, `threshold`, `left_label`, and `right_label`. print each of these and also the Gini impurity and number of training samples falling in each leaf
        * with a `.predict(X)` method that, given the `DataFrame`, `X`, containing features (only---not the class labels) of glass samples, output a list of predicted class labels for those samples.
        * your code should be general enough to work for any tabular data set pertaining to binary classification. e.g. with different features or amounts of data.

        !!! hint
            take e.g. developing the `.fit` method step-by-step by writing one line of code together with a print statement or premature `return`, then calling `decision_stump.fit(data_train[features], data_train["class"])` to make sure your code is functioning as you intend.
        """
    )
    return


@app.cell
def __(ts, y_pred):
    class DecisionStump:
        def __init__(self):
            """
            initialize a decision stump.
            (call the .fit method to assign attributes.)
            """
            print(">> initializing decision stump")
            return None

        def fit(self, X_train, y_train):
            """
            fit a decision stump to training data.
            i.e. determine the optimal feature and threshold t for the split.

            assign the following optimal decision stump parameters as attributes:
                feature, threshold, left label, right label

            X_train: pandas DataFrame containing features of training samples
            y_train: pandas Series containing associated class labels
            """
            # how much training data do we have?

            # how many features do we have?

            # initialize min impurity list and threshold list for each feature

            # outer loop over features as candidate splits
            for f, feature in enumerate(X_train.columns):
                print(f"\tconsidering split on {feature}")

                # grab unique values of this feature and sort them

                # construct list of candidate thresholds for this feature

                # inner loop over thresholds
                for t in ts:
                    t
                    # track split of data into left and right leaf nodes
                    #  under this (feature, threshold) pair

                    # compute Gini impurity in left and right leaf

                    # fraction samples left/right

                    # weighted Gini impurity of the whole split

                    # if we find a lower impurity than seen before,
                    #   store this impurity value and threshold

            # assign self.feature and self.t for the best split overall
            #   (i.e. find optimal parameters for the decision stump
            #    and store them as a class attribute for later use,
            #    particularly in the predict function.)

            # assign self.left_label and self.right_label
            #   as the majority class falling in the left and right
            #   leaf under the best split. store as a class attribute
            #   for later use in the predict function.
            #   for info, also compute purity in each leaf

        def predict(self, X):
            """
            use the trained decision stump to assign a predicted class label 
                to glass samples according to their features.all

            returns a list of class labels.

            X: pandas DataFrame containing features of glass samples
            """
            return y_pred
    return (DecisionStump,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""::noto:window:: use your implementation of `DecisionStump.fit` to train a decision stump on the training portion of the glass samples. print the optimal feature, threshold, left-leaf class label and right-leaf class label.""")
    return


@app.cell
def __(DecisionStump, data_train, features):
    decision_stump = DecisionStump()
    decision_stump.fit(data_train[features], data_train["class"])
    return (decision_stump,)


@app.cell
def __(decision_stump):
    # best feature for split
    decision_stump.feature
    return


@app.cell
def __(decision_stump):
    # associated best threshold for split
    decision_stump.t
    return


@app.cell
def __(decision_stump):
    # label for samples falling in the left leaf
    decision_stump.left_label
    return


@app.cell
def __(decision_stump):
    # label for samples falling in the right leaf
    decision_stump.right_label
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ::noto:window:: use your implementation of `DecisionStump.predict` to obtain _predicted_ class labels for the glass samples in the training set. compare to the true labels via drawing a _confusion matrix_ using `scikitlearn`'s '`ConfusionMatrixDisplay.from_predictions`. also, compute the accuracy of the decision stump on the train set.

        !!! note
            the confusion matrix and accuracy on the _train_ set is not a faithful evaluation of the decision stump, since the decision stump was "shown" the labels for the training data...
        """
    )
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
        ::noto:window:: now draw a confusion matrix and compute the accuracy using the _test_ set of glass samples.

        !!! note
            the confusion matrix and accuracy on the _test_ set is a faithful evaluation of the decision stump's _generalization error_, since the decision stump was _not_ "shown" the labels for the test data... the accuracy on the test set is _lower_ than for the train set owing to this. 

        ❓ what do you conclude about the utility of your decision stump model? particularly, comment on whether you trust the model more if it predicts a glass sample is float-processed, vs. if the glass sample is predicted to be not float-processed.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""::noto:window:: finally, use `scikitlearn` to train a decision tree classifier with `max_depth=1`, pertaining to a decision stump, to check your implementation. display the decision tree with `plot_tree`. notably, your implementation of the decision stump should have produced an identical decision stump as scikit-learn, since scikit-learn uses the Gini impurity to measure the quality of a split, too.""")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
