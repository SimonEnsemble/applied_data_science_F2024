import marimo

__generated_with = "0.9.8"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set_theme(style="whitegrid")
    return mo, np, os, pd, plt, sns


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # permutation tests for Skittles manufacturing

        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/Skittles-Louisiana-2003.jpg" alt="drawing" style="width:200px;"/>

        !!! credit
            this fun assignment is a modified version of [lab 4](https://github.com/dsc-courses/dsc80-2023-sp/blob/main/labs/04-hyp-perm/lab.ipynb) in Berkeley's course "Data 8: The Foundations of Data Science" 2023.

        Skittles are made in two locations in the United States: Yorkville, Illinois and Waco, Texas. 
        in these factories, Skittles of different colors are made separately by different machines and combined/packaged into bags for sale. 

        ::bx:data:: the tab-separated file `skittles.tsv` contains data regarding the composition (in terms of counts of red, orange, yellow, green, and purple skittles) of a sample of 468 bags of Skittles produced at the two factories.

        !!! important
            this is just a small _sample_ of the _many_ bags produced at these factories.

        we will compare the color distribution of Skittles between bags made in the Yorkville factory and bags made in the Waco factory. most people have preferences for their favorite flavor, and there is a surprising amount of variation among the distribution of flavors in each bag.

        ## data exploration

        ::icon-park-outline:candy:: read in the `.tsv` file as a pandas data frame assigned as a variable `skittles`.
        """
    )
    return


@app.cell
def __():
    # read in .tsv file


    # get list of colors


    # get list of factories


    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""::icon-park-outline:candy:: out of curiosity, what is the average total number of skittles per bag? standard deviation?""")
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
        ::icon-park-outline:candy:: create a new data frame that lists the mean number of skittles per bag of each color, among the two factories.

        | Factory | red | orange | yellow | green | purple |
        | --- | --- | --- | --- | --- | --- |
        | Waco | 12.118182 | ? | ? | ? | ? |
        | Yorkville | ? | ? | ? | ? | ? |
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
        ::icon-park-outline:candy:: create a dodged bar plot to visualize and compare the mean number of skittles per bag of each color among the two factories. color each xtick label according to the color of the Skittle. i.e. one bar per (color, factory) pair with the bars grouped by color. include an appropriate x-label, y-label, and legend.

        !!! hint
            you can use [`pd.melt`](https://pandas.pydata.org/docs/reference/api/pandas.melt.html) then [`sns.barplot`](https://seaborn.pydata.org/generated/seaborn.barplot.html) for this, or follow matplotlib's example [here](https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html).
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
        ## A/B tests on colors of skittles among the two factories

        first, we'll use a permutation test to assess whether, on average, bags made in Yorkville have the same number of `color` skittles as bags made in Waco, with `color` being "orange", "green", "red", "purple", or "yellow". the goal here is to write Python functions to do this.

        ::icon-park-outline:candy:: below, implement the following functions for such A/B tests:

        * `diff_of_means`: takes in a `DataFrame` like `skittles` and a `color` and returns the absolute difference between the mean number of `color` Skittles per bag from Yorkville and the mean number of `color` Skittles per bag from Waco.

        * `simulate_null`: takes in the original `skittles` `DataFrame` and returns one simulated instance of the data under the null hypothesis. specifically, this will involve shuffling the 'Factory' column.

        * `simulate_test_stat_under_null`: takes in the original `skittles` `DataFrame` and a `color` and returns one simulated instance of the test statistic (pertaining to that `color`) under the null hypothesis.

        * `pval_color`: takes in the original `skittles` `DataFrame` and a `color` and calculates the p-value for the permutation test using 1500 trials.

        !!! note
            looking ahead when we consider different colors, we're endowing most of our functions above with an argument `color` so they work for any color.

        !!! tip
            test each of your functions after you write them to make sure they do what you want.
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
        ## do the A/B test for orange skittles 🟠

        conduct an A/B test for the number of orange skittles per bag among the two factories. (use your functions above.)

        ::icon-park-outline:candy:: write your null and alternative hypotheses here and explain in words what the test statistic is.

        _null hypothesis_: 

        _alternative hypothesis_: 

        _test statistic_: 

        ::icon-park-outline:candy:: use 1500 trials and plot a histogram of the test statistic under the null hypothesis, along with the observed test statistic.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""::icon-park-outline:candy:: what is the p-value? what is your conclusions?""")
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
        ## do the A/B test for each color of Skittles 🔴🟠🟡🟢🟣

        ::icon-park-outline:candy:: loop through all colors and compute the p-value for each color. store the p-values in a data frame sorted by p-value to indicate which colors differ the most between the two locations on average. plot the p-values in a bar plot, with bars colored according to the Skittle color.

        💡 even though there is randomness in the color composition in each bag, this list gives the likelihood that the machines have a systematic, meaningful, difference in how they blend the colors in each bag.
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
        ## A/B testing of the overall distribution of colors among Skittles produced at the factories

        now, suppose you would like to assess whether the two locations make similar amounts of each color overall. i.e., suppose we:

          * combine and count up all the Skittles of each color in our samples that were made in Yorkville (e.g. 2917 total red skittles, 2784 total green skittles, etc.), then calculate the fraction of Skittles that were red, green, etc. 
          * do the same for the samples of bags from the Waco factory.

        ::icon-park-outline:candy:: create a data frame that contains the proportion of Skittles that are each color, broken down by factory, like below. draw a bar plot with a dodge to visualize the probability distribution of Skittles among colors, broken down by factory.

        | Factory | red | orange | yellow | green | purple |
        | --- | --- | --- | --- | --- | --- |
        | Waco | 0.204511 | ? | ? | ? | ? |
        | Yorkville | ? | ? | ? | ? | ? |
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
        r"""
        are the two distributions of the 🏭's Skittle colors similar? is the variation among the bags (studied above) due to each factory making different relative amounts of each color (studied here)?

        ::icon-park-outline:candy:: use a permutation test to assess whether the distribution among colors of Skittles made in Yorkville is statistically significantly different than those made in Waco. set a significance level (i.e. p-value cutoff) of 0.01 and determine whether you can reject a null hypothesis that answers the question above using a permutation test with 1500 trials. for your test statistic, use the total variation distance (TVD) described [here](https://inferentialthinking.com/chapters/11/2/Multiple_Categories.html#).

        some guidance:

          * our previous permutation tests have compared the mean number of (say) orange Skittles in Yorkville bags to the mean number number of orange Skittles in Waco bags. The role of shuffling was to randomly assign bags to Yorkville and Waco.
          * in this permutation test, we are still shuffling to randomly assign bags to Yorkville and Waco. the only difference is that after we randomly assign each bag to a factory, we will compute the distribution among colors between the two factories and compute the TVD between those two distributions.
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ::icon-park-outline:candy:: write out the null and alternative hypothesis for this permutation test below. do you reject the null hypothesis?

        _null hypothesis_:

        _alternative hypothesis_:

        I [accept/reject] the null hypothesis under a significance level of 0.01.
        """
    )
    return


if __name__ == "__main__":
    app.run()
