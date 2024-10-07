import marimo

__generated_with = "0.8.20"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # wrangling with tabular data

        to read, store, query, filter, sort, group, manipulate, etc. tables of data, we will employ the [pandas](https://pandas.pydata.org/) library.
        > pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
        built on top of the Python programming language.

        💡 a [`DataFrame`](https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html) is a two-dimensional data structure (in many languages) for storing data in tabular form. entries can be different data types, like floating point values, integers, and strings. it's like a spreadsheet. 

        * rows = instances/examples
        * columns = features of those instances/examples

        ## simple operations on `DataFrame`'s

        ### construct a `DataFrame` from scratch

        e.g. a data frame cataloging different bottles of wine in our possession.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### insert a new column""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### get the column names, number of rows, number of columns""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### grab a column

        retreiving a column from a data frame works like a dictionary and returns a `Series`.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### append rows""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### transform a column

        e.g. convert alcohol per volume (APV) `Series` from a percentage to a fraction.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### rename a column

        _note_: `inplace=True` will modify the `DataFrame`, `df`, while the below just returns a copy.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### select a subset of columns

        suppose we're only interested in the type and variety of the wine.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### filter the rows""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### grab a value

        what's the acidity of the Chardonnay?
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### drop columns or rows

        _note_: `inplace=true` will modify the `DataFrame` in place. the code below just returns a copy.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""### sorting""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## grouping (split-apply-combine)

        i.e. [split-apply-combine](https://pandas.pydata.org/docs/user_guide/groupby.html).

        > A groupby operation involves some combination of splitting the object, applying a function, and combining the results. This can be used to group large amounts of data and compute operations on these groups.

        ### split
        e.g. group by type of wine (red/white) and iterate through the groups.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""get a certain group.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### split-apply-combine

        apply a function to each group, then combine the result.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## missing values

        first, introduce missing values. `np.NaN` for `float`s.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""e.g. count missing values within each group.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""replace the missing values with something different.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## reading/saving

        write to a `.csv` (comma-separate value) file.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""read from a `.csv` file.""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### concatenating/joining data tables""")
    return


if __name__ == "__main__":
    app.run()
