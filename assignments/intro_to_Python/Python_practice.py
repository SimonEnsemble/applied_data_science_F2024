import marimo

__generated_with = "0.8.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # modeling and visualizing the trajectory of a projectile 

        ⚾ write a [`class`](https://docs.python.org/3/tutorial/classes.html) `Projectile` that models the vertical ($y$-direction) and horizontal ($x$-direction) displacement of a projectile using the kinetic equations [here](https://en.wikipedia.org/wiki/Projectile_motion#Displacement). 

        endow the `Projectile` class with two _attributes_: 
        (i) the magnitude [m/s] of its initial velocity and
        (ii) the direction of its initial velocity vector, described by its angle [radians] with the x-axis. 
        (assume the projectile begins, at time $t=0$, at the origin $(x=0, y=0)$ which is a point on the ground. so, you don't need to store the initial position.)

        endow the `Projectile` class with the _methods_ that enable:

        * computing and returning the $x$ position of the projectile at arbitrary time $t$
        * computing and returning the $y$ position of the projectile at arbitrary time $t$
        * computing and returning the time of flight of the projectile
        * computing and returning the horizontal range of the projectile
        * computing and returning the maximum height reached by the projectile
        * visualizing the historical path and current position of the projectile at arbitrary time $t$ using [`matplotlib`](https://matplotlib.org/). this method should call some of the previously coded methods.
            * enforce equal scaling on the $x$- and $y$ axes to give physical perspective about the shape of the trajectory
            * impose fixed $x$ limits, from zero to 110% of the horizontal range.
            * impose fixed $y$ limits, from zero to 110% of the maximum height reached.
            * label the x- and y-axes and indicate units in the label.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""⚾ construct an instance of your `Projectile` class, having initial (at $t=0$) velocity of magnitude $v_0=20$ m/s and angle $\theta=\frac{4}{9}\pi$ with the $x$-axis.""")
    return


@app.cell
def __(Projectile, np):
    projectile = Projectile(20.0, np.pi * 4 / 9)
    return (projectile,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("""⚾ what is the $x$- and $y$-coordinate [m] of the projectile at $t=1$ s? print the positions to two decimal places.""")
    return


@app.cell
def __(mo):
    mo.md("""⚾ what is the time of flight [s], horizontal range [m], and maximum height [m] of the projectile? print to two decimal places.""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""⚾ visualize the historical path and current position of the trajectory at $t=1$ s.""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""⚾ to visually scan the historical path and current position of the trajectory at a range of times spanning from zero to the flight time, implement a slide for the time variable using `marimo`'s [`slider`](https://docs.marimo.io/api/inputs/slider.html) function then pass the value of time contained in the slider to your visualization method of the `projectile` object. slide the value of time from $t=0$ until the end of flight (when the projectile hits the ground) and check that the trajectory makes sense, and that the plot axes are fixed so it looks like a video in a fixed frame.""")
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
        ## automating the reading of data from files output from an instrument

        often, an instrument in a lab outputs a single file per analysis of a sample.
        then, it is advantageous to have a `class` that reads the data from such a file and populates attributes of the sample and measurements. the following problem exemplifies.

        🌱 the files `sample_*.txt` contain information about (triclinic) crystal structures of different metal-organic frameworks (MOFs) from the supporting information of a publication ([link](https://pubs.acs.org/doi/10.1021/ja500330a)). 
        create a class `Crystal` whose constructor, whose argument is an integer `i`, reads from the file `sample_{i}.txt` (i) the crystal lattice lengths `a`, `b`, and `c`, (ii) the crystal lattice angles `alpha`, `beta`, and `gamma`, and (iii) MOF name, and stores them as attributes. also create two methods for the `Crystal` class, one that computes and returns the volume of the crystal lattice (see the formula for the volume of a triclinc system [here](https://en.wikipedia.org/wiki/Bravais_lattice#In_3_dimensions)) and another that prints the name of the MOF, the values of the lattice parameters with their units, and the volume of the lattice with the appropriate units.
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
    mo.md("""🌱 construct an instance of your class on `sample_5.txt` and call the method that prints the attributes it read and computed volume.""")
    return


@app.cell
def __(Crystal):
    crystal = Crystal(5)
    return (crystal,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
