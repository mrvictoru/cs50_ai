import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains:
            for word in self.domains[v]:
                if v.length != len(word): # constrain condition, word lenght need to be the same as variable length
                    self.domains[v].remove(word) # remove node for failing constrain


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlaps = self.crossword.overlaps[x,y]
        ret = False

        # return false for no overlaps as no revision was made
        if overlaps is None:
            return ret
        
        # to check whether the word in x and word in y has the same character at overlap position
        for xword in self.domains[x].values():
            for yword in self.domains[y].values():
                if xword[overlaps[0]] != yword[overlaps[1]]:
                    self.domains[x].remove(xword)
                    ret = True
        
        return ret


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = []
            # loop through all var in domain
            for x in self.domains:
                # loop through all neighbor of x
                for neighbor in self.crossword.neighbors(x):
                    # finding neighbors of x and pair them as an arc
                    arc = (x,neighbor)
                    # if such arc not already in list, add them
                    if arc not in arcs:
                        arcs.append(arc)
        
        # as long as the arcs list isnt empty
        while not arcs:
            # dequeue from arcs
            arc = arcs.pop()
            # to enforce arc consistant and check whether it happened 
            if self.revise(arc[0],arc[1]):
                # if nothing left in arc[0], no condition satsify csp
                if len(arc[0]) == 0:
                    return False
                # add neighbor back in as it might no longer be arc consistant
                for z in self.crossword.neighbors(arc[0]):
                    if z != arc[1]:
                        arcs.append((z,arc[0]))

        return True               

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # loop through the list of words in assignment
        for words in assignment.values():
            # if there is more than 1 word in the list, assignment not complete
            if len(words) != 1:
                return False
            else:
                return True
            

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        ret = True
        # loop through the assignment
        for variable in assignment:
            # check for whether all values are distinct
            if len(assignment.variable) > len(set(assignment.variable)):
                ret = False
            # loop through all the words in each variable
            for words in assignment[variable]:
                # check for node consistancy
                if len(words) != variable.length:
                    ret = False
            # loop through neighbor of variable to check for conflict
            for neighbor in self.crossword.neighbors(variable):
                overlaps = self.crossword.overlaps[variable,neighbor]
                for vword in self.domains[variable].values():
                    for neword in self.domains[neighbor].values():
                        if vword[overlaps[0]] != neword[overlaps[1]]:
                            ret = False

        return ret

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        orderlist = []
        # loop through the values in those domain
        for values in var.values():
            cell = ()
            n = 0
            # loop through the values in neigbor's domain
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                # count whether how many same values in the the neighbor.
                    if values == self.domains[neighbor]:
                        n += 1
            cell = (values,n)
            orderlist.append[cell]
        # sort the list with the least number of values among the neighbors to most
        orderlist.sort(key = lambda x:x[1])        
        return orderlist

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        remainlist = []
        # loop through the domain
        for variable in self.domains:
            # check if it is assigned
            if variable not in assignment:
                cell = (variable,len(self.domains[variable]))
                # put the variable and num of remaining values in list
                remainlist.append[cell]
        # sort according number of remaining values
        remainlist.sort(key = lambda x:x[1])
        
        # take
        while True:
            if remainlist[-1][1] == remainlist[0][1] and len(remainlist) > 1:
                break
            elif len(remainlist) == 1:
                return remainlist[0]
            else:
                remainlist.pop()

        

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        raise NotImplementedError


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    print("check")

    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)



if __name__ == "__main__":
    main()
