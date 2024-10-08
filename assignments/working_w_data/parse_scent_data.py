import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    from rdkit import Chem
    import os
    return Chem, Draw, mo, np, os, pd, plt


@app.cell
def __(plt):
    plt.style.use('bmh')
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"# {mo.icon('twemoji:nose')} wrangling with odorant perception data")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        !!! objective

        pre-process, combine, and analyze the odorant perception data of _Goodscents_ and _Leffingwell_, obtained from [_The Pyrfume Project_](https://pyrfume.org/). 

        loosely, each instance in this tabular data represents the outcome of an experiment where we expose a human olfaction expert to a pure compound and ask them what it smells like.

        we will combine this data as in ["Machine Learning for Scent: Learning Generalizable Perceptual Representations of Small Molecules"](https://arxiv.org/abs/1910.10685), who trained a machine learning model to predict the human olfactory perception of molecules from their chemical structures. (more at the [Google AI blog](https://ai.googleblog.com/2019/10/learning-to-smell-using-deep-learning.html) and [a follow-up study](https://www.science.org/doi/10.1126/science.ade4401)).

        !!! note
        I wrote a Python script `get_data.py` that employs the `pyrfume` package to query the raw data from Goodscents and Leffingwell. both the Python script and the resulting `.csv` files are on the Github repo for our course.

        !!! note

        throughout, we will use the [simplified molecular-input line-entry system (SMILES)](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) to specify the structure of the molecule in the olfaction data.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"## {mo.icon('twemoji:nose')} read in and join the Leffingwell data")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        read in the two `.csv` files of the Leffingwell data set, join the tables, then manipulate the joined data frame so it looks as follows:

        | molecule	| lw-labels |
        | ----	| ----- |
        | CCCCC=COC(=O)CCCCCCCC | [green, oily, fruity, waxy, herbal] |
        | ... | ... |
        | CCC1CSSSC1(CC)CC | [green, alliaceous, savory, onion] |

        particularly, the column with the olfactory perception labels should be a list of strings.

        !!! hint

        you'll probably need to use the `rename`, `transform`, `replace`, `merge`, and  `split` functions.
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
    # merge

    # string replacements

    # split string into list of olfactory labels

    # rename cols

    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} how many _unique_ molecules are in the Leffingwell data set? does each row in the data frame pertain to a unique molecule?")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} how many _unique_ olfactory perception labels (sweet, herbal, etc.) are in the Leffingwell data set?")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} what is the maximum number of labels a molecule in the Leffingwell data set has? the minimum? the average?")
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
    mo.md(f"## {mo.icon('twemoji:nose')} read in and join the Goodscents data")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        read in the three `.csv` files of the Goodscents data set, join the tables, then manipulate the joined data frame so it looks as follows:

        | molecule | gs-labels |
        | ---- | ---- |
        | CC(=O)C1=CC=C(C=C1)OC	| [sweet, vanilla, cherry maraschino cherry, powdery, anisic, balsamic, hawthorn, acacia] |
        | ... | ... |
        | CC[C@H](C)CCC=O | [bland] |

        again, the column with the olfactory perception labels should be a list of strings.

        !!! hint

        you'll probably need to use the `rename`, `transform`, `dropna`, `merge`, and  `split` functions.
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
    # successively merge the three tables

    # rename columns

    # drop rows with missing labels

    # convert string labels to list of labels

    # turn labels into list

    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} how many _unique_ molecules are in the Goodscents data set? does each row in the data frame pertain to a unique molecule?")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('emojione-monotone:face-screaming-in-fear')} the molecules listed in the rows are not unique! let us investigate then address this.")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} use the `duplicated` function [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html) to obtain all rows of the dataframe involving a molecule that is dupilicated (keep _all_ duplicates, not just one of them). note, some of the duplicated molecules actually have different labels. so it would be irresponsible to simply drop one of the duplicates (e.g. the first or second one). this would constitute discarding valuable and expensive-to-obtain data! {mo.icon('emojione:dizzy-face')}")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} to make the Goodscents data frame have unique molecules in the rows, merge the labels of duplicate molecules.")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        so, for example, the two rows pertaining to the duplicate molecule:

        | | |
        | --- | --- |
        | CCC1CCC2(CC1)CCC(=O)O2| [creamy, peach, lactonic] |
        | CCC1CCC2(CC1)CCC(=O)O2 | [creamy, spicy, cinnamon, lactonic] | 

        should be merged to constitute one row:

        | | |
        | --- | --- |
        | CCC1CCC2(CC1)CCC(=O)O2 | [creamy, peach, lactonic, spicy, cinnamon] |

        !!! hint

        `groupby`, `agg`, list concatenation.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} now, what is the maximum number of labels for a molecule in the Goodscents data set? the average?")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} how many _unique_ olfactory perception labels (sweet, herbal, etc.) are in the Goodscents data set?")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"## {mo.icon('twemoji:nose')} join the Goodscents and Leffingwell data")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        join the Goodscents and Leffingwell data frames. the final data frame should look like:


        | molecule | labels |
        | ---- | ---- |
        | CC(=O)C1=CC=C(C=C1)OC	 | [powdery, hawthorn, hay, balsamic, cherry maraschino cherry, vanilla, acacia, anisic, floral, sweet] | 
        | ... | ... |
        | C(CSCCC(S)S)C(S)S	| [fruity, alliaceous, green, onion, garlic] | 

        again, the column with the olfactory perception labels should be a list of strings.

        * for molecules in both Leffingwell and Goodscents, the labels should be the concatenated list of labels from both data sets. only keep the unique ones. i.e. labels should not be repeated on the same molecule.
        * some molecules are present in the Goodscents data only, and some are present in the Leffingwell data only; _both_ sets of these molecules should be included in this data frame. i.e. do not drop molecules that appear in Leffingwell but not Goodscents and vice versa.
        * many of the odor labels should be merged, e.g. "eggy" and "egg", since they mean the same thing, arguably. conduct all odor label replacements present in the file `odor_label_replacements.csv`, within a for loop.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    # join Goodscents and Leffingwell

    # convert missing to an empty list

    # combine labels

    # only keep unique labels

    # keep only molecule and labels

    # odor label replacements

    # insert counts of odor labels

    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} how many _unique_ molecules are in the combined data set? (this should match the number of rows.)")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"## {mo.icon('twemoji:magnifying-glass-tilted-left')} analysis of the combined data set")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} how many _unique_ olfactory perception labels (sweet, herbal, etc.) are in the combined data set?")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} to check your odor label replacement, ensure that e.g. rubbery has been replaced with rubber by checking it isn't a label in the data anymore.")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} append a new column to your data frame that lists the number of unique olfactory perception labels on each molecule in the data. draw a bar plot showing the number of odor labels per molecule in the combined data set. the bar heights should show the counts of molecules with 1 odor label, 2 odor labels, and so on. put xticks at 1, 2, ...")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        !!! hint

        you can get the number of molecules with each number of odor labels with a `groupby` and applying a `count` function in a split-apply-combine strategy. use this to draw the bar plot.
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
    mo.md(f"{mo.icon('codicon:question')} filter the rows of the data frame pertaining to molecules that smell like eucalyptus.")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} use RDKit (see [here](https://www.rdkit.org/docs/GettingStartedInPython.html)) to draw the structure of the only molecule that smells like both eucalyptus and grassy. (first, filter the rows to obtain this molecule.)")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} construct a new data frame that lists the odor labels and counts the number of molecules having that label. sort the odor labels by prevalence in descending order.")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        i.e., your data frame should look like:

        | label |	# molecules |
        | ---- | ---- | 
        | fruity | 2052 |
        | green	| 1599 |
        | ... | ... |
        | amascone	| 1 |
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} make a bar plot showing the number of molecules per odor label, for the top 30 most prevalent labels. the y-axis should list the 30 most prevant odor labels in order, then the bars stretching horizontally should indicate the number of molecules with that label. change the size of the figure so the labels don't overlap. remove any awkward white gaps with `ylim`.")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"{mo.icon('codicon:question')} (thinking of some hierarchy in the odor labels) what fraction of molecules with the odor label apple are _also_ labeled as fruit?")
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
