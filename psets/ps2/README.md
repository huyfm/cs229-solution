# CS229 Problem Set Instructions


## Setup for Written Parts

1. We have provided a LaTeX template in the `tex/` directory to make it easy to typeset your homework solutions.
2. Every problem has its own directory (*e.g.,* `tex/featuremaps/` for Problem 1).
3. Every subproblem has two files within the parent problem’s directory:
  - The problem statement, *e.g.* `tex/featuremaps/01-degree-3-math.tex` for Problem 1(a)). You do not need to modify this.
  - Your solution, *e.g.* `tex/featuremaps/01-degree-3-math-sol.tex` for your solution to Problem 1(a). You will need to modify these files (and the source files in `src` for coding parts).
4. You can use the given `Makefile` to typeset your solution, or use an editor with built-in typesetting such as TeXShop (comes free with the standard [LaTeX distribution](https://www.latex-project.org/get/)) or [Texpad](https://www.texpad.com/) (separate download, not free).


## Setup for Coding Parts

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. Extract the zip file and run `conda env create -f environment.yml` from inside the extracted directory.
  - This creates a Conda environment called `cs229`
3. Run `source activate cs229`
  - This activates the `cs229` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/cs229/bin/python`
  - Select `OK` then `Apply`
5. Notice some coding problems come with `util.py` file. In it you have access to methods that do the following tasks:
  - Load a dataset in the CSV format provided in the problem
  - Add an intercept to a dataset (*i.e.,* add a new column of 1s to the design matrix)
  - Plot a dataset and a linear decision boundary. Some plots might require modified plotting code, but you can use this as a starting point.
7. Notice that start codes are provided in each problem directory (e.g. `gda.py`, `posonly.py`)
  - Within each starter file, there are highlighted regions of the code with the comments ** START CODE HERE ** and ** END CODE HERE **. You are strongly suggested to make your changes only within this region. You can add helper functions within this region as well.
8. Once you are done with all the code changes, run `make_zip.py` to create a `submission.zip`.
  - You must upload this `submission.zip` to Gradescope.
