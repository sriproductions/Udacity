assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

# diagonal sudoku -- add additional diagonal constraint units
diag_unit1 = [[rows[i]+cols[i] for i in range(0,len(rows))]]
diag_unit2 = [[rows[i]+cols[8-i] for i in range(0,len(rows))]]

#list of all units in puzzle
unitlist = row_units + column_units + square_units + diag_unit1 + diag_unit2

# dictionaries of units and peers associated with a box
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

#======= implementation of two out of three strategy as outlined in sudokudragon.com
def create_work_tuples(tripletlist, rcunitlist):
    """
    given a set of 3 rows or three colums, generate a set of tuples that can
    be scanned to determine if we can apply two-out-of-three strategy to this set of
    three rows/columns.
    Input Args:
        tripletlist: either [[A1,A2,A3][A4,A5,A6][A7,A8,A9]] (columns)
                            [[A1,B1,C1],[D1,E1,F1],[G1,H1,I1]] (rows)
                            Each item in the triplet identifies a row or a columns -- it is
                            the first box in a row/column
        rcunitlist:  list of rows if triplet corresponds to rows or list of columns if triplet
                    corresponds to columns
    Output Args:
        for a triple [box1,box2,box3] create a list of tuples like ([box1,box2], box3)
        for all possible combinations of picking 2 out of these three colums/rows
        i.e, for [b1,b2,b3]: return [([b1,b2],b3),([b1,b3],b2),([b2,b3],b1)]
        returns a list of these tuples of all items in tripletlist
    """
    tuplelist = []
    for u in tripletlist:
        # pairlist is list of all possible pairs from triplet23
        # for triplet [A1,A2,A3] generate all possible pairs in a list [[A1,A2],[A1,A3],[A2,A3]]
        pairlist = [[u[i],u[j]] for j in range(0,3) for i in range(0,3) if j  > i]
        for pair in pairlist:
            box3 = [b for b in u if b != pair[0] and b != pair[1]]
            # insert all tuples in a single list
            t = pair,box3[0]
            tuplelist.append(t)
    return tuplelist

def two_out_of_three(values):
    """
    two out of three strategy: pick three colums at a time (so that they comprise three box units)
    search for values assigned to exactly two out of three rows. That leaves exactly three boxes
    in the remaining row, where the value assigned to the other two rows can fit

    Arg:
        values(dictionary) of sudoku boxes
    Used the following board given in sudokudragon.com to test solution
    sudoku_23='..951..62634...59.1256397.425.84763.46..5..17.87361.255.6173248.12...97674..961..'

    """
    new_values = values.copy()
    # create lists of row groups and column groups
    cols23 = [['A'+cols[3*i+j] for j in range(0,3)] for i in range(0,3)]
    coltuples = create_work_tuples(cols23, column_units)
    rows23 = [[rows[3*i+j]+'1' for j in range(0,3)] for i in range(0,3)]
    rowtuples = create_work_tuples(rows23, row_units)
    a = 0
    for tuplelist in [rowtuples,coltuples]:
        if tuplelist == rowtuples:
            unitslist = row_units
        else:
            unitslist = column_units
        for t in tuplelist:
            pair = t[0]
            box3 = t[1]
            box1_unit = [u for u in unitslist if pair[0] in u]
            box2_unit = [u for u in unitslist if pair[1] in u]
            box3_unit = [u for u in unitslist if box3 in u]
            # scan box1_unit and box2_unit boxes for matching single choice
            for box1 in box1_unit[0]:
                if len(values[box1]) > 1:
                    continue
                for box2 in box2_unit[0]:
                    if values[box2] == values[box1]:
                        # we have single choice values[box1] in box1_unit and box2_unit now,
                        # skip boxes in box3_col are not in the box3 square unit
                        # leaving three remaining boxes, one of which we can
                        # potentiall fill with values[box1].
                        #                               However, we can only do
                        # that if 2 out of the three boxes have a single choice

                        # pick the three square units associated with the columns/rows of interest
                        trip_sq_unit = []
                        for sq in box1_unit[0]:
                            for squnit in square_units:
                                if sq in squnit and squnit not in trip_sq_unit:
                                    trip_sq_unit.append(squnit)
                        assert(len(trip_sq_unit) == 3)
                        box3_sq_unit = [squnit for squnit in trip_sq_unit if box1 not in squnit and box2 not in squnit]
                        many_choice_boxes = [b for b in box3_unit[0] if b in box3_sq_unit[0] and len(values[b]) > 1]
                        if (len(many_choice_boxes) == 1):
                            # first check that no peer of this box already has values[box1]
                            for sq in peers[many_choice_boxes[0]]:
                                if values[sq] == values[box1]:
                                    break
                            else:
                                assign_value(new_values,many_choice_boxes[0],values[box1])
                else:
                    # no match for values[box1] in box1_unit or box2_unit
                    continue
    return new_values

#======

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    ntwins=[] # list of (box, unit) tuples
    # Find all instances of naked twins
    for unit in unitlist:
        for sq in unit:
            # ignore boxes that cannot be a naked twin choice
            if len(values[sq]) != 2:
                continue
            # scan for naked twin in unit
            for maybe_twin in unit:
                if maybe_twin != sq and values[maybe_twin] == values[sq]:
                    #insert tuple
                    t = sq,unit
                    ntwins.append(t)
    # ntwins contains a list of (box, unit) tuples
    # walk down tuple list and apply naked_twin elmination from boxes that
    # do not contain naked twin
    for tup in ntwins:
        sq = tup[0]
        unit = tup[1]
        for u in unit:
            #  check for len(values[u] > 1 maybe not be required
            if len(values[u]) > 1 and values[u] != values[sq]:
                v = values[u]
                # replace naked twin choices from box
                for c in values[sq]:
                    v = v.replace(c,"")
                # assign value if anything changed
                if v != values[u]:
                    assign_value(values,u,v)
    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = []
    all_digits = '123456789'
    assert len(grid) == 81
    for c in grid:
        if c == '.':
            values.append(all_digits)
        else:
            values.append(c)
    return dict(zip(boxes, values))


def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    if values is False:
        print("No Solution")
        return

    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    """

    Eliminate choice x from a box if there a peer of that box with a single choice x
    Args:
        values(dict): the sudoku in dictionary form
    """
    # not necessary to make a copy, but it is easier to debug if original is not changed
    new_values = {}
    # iterate through every box and remove any choice that is already
    # in a peer box.
    for sq in values:
        choices = values[sq]
        for peer in peers[sq]:
            if (len(values[peer]) == 1):
                choices=choices.replace(values[peer],"")
        # do not update values but update a copy to assist in debugging
        assign_value(new_values,sq,choices)
    return new_values


def only_choice(values):
    """

    only_choice strategy: if a choice fits only one box in a unit, assign it.
    make a copy of values dict otherwise the order of scanning results in
    different (and wrong) results
    Args:
        values(dict): the sudoku in dictionary form
    """
    new_values = values.copy()  # note: do not modify original values
    for unit in unitlist:
        for c in '123456789':
            choices=[]
            for sq in unit:
                if c in values[sq]:
                    choices.append(sq)
            if len(choices)==1: #only one match for the character in unit
                assign_value(new_values, choices[0], c)

    return new_values


def reduce_puzzle(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Use the Eliminate Strategy
        values = eliminate(values)
        # Use the Only Choice Strategy
        values = only_choice(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    #"Using depth-first search and propagation, create a search tree and solve the sudoku."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    # not a valid solution, return False
    if values is False:
        return False
    # are we done? yes, if all boxes have only one choice
    for s in values.keys():
        if len(values[s]) > 1:
            break
    else:
        # we are done
        return values
    # Choose one of the unfilled squares with the fewest possibilities
    min=9
    choice=""
    for s in values.keys():
        if len(values[s]) > 1 and len(values[s]) < min:
            choice=s
            min=len(values[s])
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False),
    # return that answer!
    for c in values[choice]:
        new_values=values.copy()
        assign_value(new_values, choice, c)
        result=search(new_values)
        if result is not False:
            return result
    # If you're stuck, see the solution.py tab!
    return False

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    # convert to dictionary and replace . with 123456789
    values = grid_values(grid)
    # apply eliminate and only_choice
    values = reduce_puzzle(values)
    # Use Naked Twins Strategy
    values = naked_twins(values)
    # use two out of three strategy
    values = two_out_of_three(values)
    #apply depth first search to find solution, if it exists
    values = search(values)
    return values

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    values=grid_values(diag_sudoku_grid)
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
