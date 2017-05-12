# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Constraint propagation applies constraining rules to a given problem
   to reduce the search space for the the solution. In the Sudoku problem, naked
   twins exploits the fact that if two boxes can only take one of the same two values,
   it contrains the remaining boxes in that unit from take either of those two values.

   Given a particular unit, we scan every entry that has exactly two choices, and see
   if another box in that unit has one of these two choices. If so, then we remove those
   choices from the rest of the boxes in the unit, since we know that no other square in
   that unit can take the values associated with the twin boxes. For example, if two
   squares in a unit have 23 as possible values, then we remove 2 and 3 as possible
   choices for all other squares in that unit.

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: With Diagonal Sudoku, two additional constraints are applied by requiring that
   both diagonals of a board also satisfy the criterion of having each digit exactly
   once.  This results in eliminating choices from squares that would not have been
   eliminated without these additional constraints.

   Programatically, Diagonal Sudoku is regular Sudoku with two
   additional diagonal units in the unit list.

NOTE: I have also implemented two-out-of-three as explained in sudokudragon.com

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.