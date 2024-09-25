import marimo

__generated_with = "0.8.18"
app = marimo.App(css_file="cory.css")


@app.cell
def __():
    import marimo as mo
    import math
    import os
    from dataclasses import dataclass
    return dataclass, math, mo, os


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # introduction to Python

        a list of topic's we'll cover, through example tasks:

         * variable assignment and printing
         *  functions
         *  lists
         *  dictionaries
         *  classes

        [here](https://docs.python.org/3.12/tutorial/index.html) is a link to Python's official tutorial.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## numbers, variable assignments, and printing
        🐸 what's the volume of a sphere of radius 2 cm?
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
        ## lists

        🐸 write out some terms of the [Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_sequence) in a list.
        """
    )
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""check the length.""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""indexing and slicing""")
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
    mo.md(r"""iterate through the list (three ways).""")
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
    mo.md("""🐸 create a list of numbers $0, 1, ..., 9$ (ten of them).""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""🐸 use a list comprehension to create a list of odd numbers 1, 3, ..., 23.""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## dictionaries

        they store (key, value) pairs.

        🐸 create a dictionary that maps atom types to their atomic mass.
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## functions
        🐸 write a function that computes the [truncated] [series approximation](https://en.wikipedia.org/wiki/Sine_and_cosine#Series_and_polynomials) of $\sin(x)$.

        \[
            \sin (x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} + \cdots.
        \]
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
    mo.md("""how many terms are needed to match Python's implementation of $\sin(1.2)$ with an absolute tolerance of `1e-12`?""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## classes

        🐸 create a data class to store information about different varieties of peppers.
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
    mo.md("""🐸 write a function that takes in a pepper and spice tolerance, then indicates if a person can handle eating it.""")
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
    mo.md(r"""🐸 create a class to represent a [van der Waals gas](https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Atomic_and_Molecular_Properties/A8%3A_van_der_Waal%27s_Constants_for_Real_Gases).""")
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
        ## reading in a file

        🐸 read in the text file "density_of_common_liquids.txt" and automatically populate a dictionary that maps a liquid to its [mean] density at 20 deg C and 1 atm.
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


if __name__ == "__main__":
    app.run()
