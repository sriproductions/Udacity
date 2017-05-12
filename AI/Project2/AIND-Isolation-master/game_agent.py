"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math
from isolation import Board
import pdb
from sample_players import RandomPlayer, GreedyPlayer

#logging.basicConfig(level=0)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def heuristic1(game,player):
    """ Heuristic 1: Heuristic picks node with the largest ratio of player moves to opponent moves
    Parameters:
    ----------
    game: isolation.Board
         an instance of islation.Board encoding current state of game

    player: callable object
         A player instance in the current game, i.e., active or inactive player

    Returns:
    -------
    float
       The number of legal moves available to the player in the input args
    """

    myscore = float(len(game.get_legal_moves(player)))
    oppscore = float(len(game.get_legal_moves(game.get_opponent(player))))
    if oppscore == 0.0:
        return "inf"
    return myscore/oppscore

def heuristic2(game,player):
    """ Heuristic 1: simple heuristic that picks the move with the largest
    value

    Parameters:
    ----------
    game: isolation.Board
         an instance of islation.Board encoding current state of game

    player: callable object
         A player instance in the current game, i.e., active or inactive player

    Returns:
    -------
    float
       The number of legal moves available to the player in the input args
    """
    mymoves = game.get_legal_moves(player)
    if len(mymoves) == 0:
        return float("-inf"); #player loses
    oppscore = float(len(game.get_legal_moves(game.get_opponent(player))))
    if oppscore == 0.0: #player wins and opponent loses
        return float("inf")
    best_score = float("-inf")
    for move in mymoves:
        my_next_score = len(game.forecast_move(move).get_legal_moves(player))
        opp_next_score = len(game.forecast_move(move).get_legal_moves(game.get_opponent(player)))
        best_score = max(best_score, (len(mymoves)*my_next_score)-opp_next_score)
    #return float(best_score)*float(len(mymoves))/oppscore
    return float(best_score)

def heuristic3(game,player):
    """ Heuristic 3: Variation of the algorithm in the paper by Ronald Rivest suggested in the project.
    Note: value of p in weight computation, maxp is set to 10 for maximing player
          and minp to 0.1 for minimizing player
    Heuristic works as follows:
    1. find all legal moves M for player L board B for player P at position p
    2. for each legal move m in M, apply m to Board B to yield B' with opponent Q in position p'
    2a     find the available moves M' for opponent in B' from position p'
    2b     compute score S and S' for player and opponent, respectively, using heuristic 1
    2c     determine the weight/penalty for edge using the negative log derivate as suggested
           in page 87,  w(p,p') = -log(len(M)) + (minp-1) * (log(S') - log(S))
    2d     for each m' in M', apply m' to board B' to yield B" so that P is at position p"
    2d.1       find legal moves M" for player P in B"
    2d.2       compute score S" for P using heuristic 1
    2d.3       determine weight(p',p") = -log(len(M')) + (naxp-1) * (log(S") - log(S'))
    3      determine min(weight(p,p')+weight(p'+p")) over all m->m'->m" sequences
    Parameters:
    ----------
    game: isolation.Board
         an instance of islation.Board encoding current state of game

    player: callable object
         A player instance in the current game, i.e., active or inactive player

    Returns:
    -------
    float
       The number of legal moves available to the player in the input args
    """
    # calculate score for this node to compute the weights/penalty
    mymoves = game.get_legal_moves(player)
    oppmoves = game.get_legal_moves(game.get_opponent(player))
    if len(oppmoves) == 0:
        return float("-inf")
    myscore = float(len(mymoves))/float(len(oppmoves))
    # for each move forecast subsequent oppostion moves and find the minx
    # and max values for the forecasted moves
    cumulative_minpenalty = float("inf")
    cumulative_minnode = (-1,-1)
    for move in mymoves:
        nextboard = game.forecast_move(move)
        forecast_oppmoves = nextboard.get_legal_moves()
        forecast_mymoves = nextboard.get_legal_moves(player)
        if  len(forecast_mymoves) == 0:
            continue
        next_score = float(len(forecast_oppmoves))/float(len(forecast_mymoves))

        minpenalty = float("inf")
        minweightnode = (-1,-1)

        if len(forecast_oppmoves) == 0 or len(forecast_mymoves) == 0:
            break

        maxp_value = 10 #maximizing node so set it to large number
        # weight = log(n) + (p-1)*(log(score(child))-log(score(parent)))
        linkweight = math.log(len(forecast_mymoves)) + \
                     (maxp_value - 1) * (math.log(next_score) - math.log(myscore))
        for oppmove in forecast_oppmoves:
            if len(forecast_oppmoves) == 0:
                break    # this is a terminal node that cannot be expanded, so skip it
            # score of child node
            my_nextscore = float(len(nextboard.forecast_move(oppmove).get_legal_moves()))/float(len(forecast_oppmoves))
            if my_nextscore > 0 and next_score > 0 and len(forecast_oppmoves) != 0:
                minp_value = 0.1
                # calculate weight of child edge given my_nextscore and oppscore are non-zero
                # weight = log(n) + (p-1)*(log(score(child))-log(score(parent)))
                weight = math.log(len(forecast_oppmoves)) + \
                         (minp_value - 1)* (math.log(my_nextscore) - math.log(next_score))
                #
                minpenalty = min(minpenalty,weight + linkweight)
                cumulative_minpenalty = min(minpenalty,cumulative_minpenalty)
    # return inverse of minweight so that lower penalty paths get highest score
    return float(1.0/math.exp(cumulative_minpenalty))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    score = game.utility(player)
    # utility returns 0 if game there is no winner, return winner/loser score otherwise
    if score != 0:
        return score

    return heuristic1(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.moves_completed = 0
        random.seed()

    def get_init_move(self,game, legal_moves):
        """ return all the squares in the part of the board given a
            position on the board
        Parameters
        ----------
        game: board

        legal_moves: legal moves for active player on board

        Returns
        -------
        set of Initial moves on board
        """

        w = game.width  # cols
        h = game.height # rows

        # start at the centre of the board as that always has maximum options
        # with respect to freedom to move to any quadrant. if that is already occupied by
        # adversary, pick an adjacent diagonal square which maximizes options for player
        if (h/2, w/2) in legal_moves:
            return [(h/2, w/2)]
        # if center square is not availablem exploit symmetry of board
        # (vertical, horizontal, and diagonal) to pick the squares
        # diagonally adjacent and alongsize (same row or column)
        # all other adjacent square choices are equivalent to these two.
        return [(h/2+r,h/2+c) for r,c in range(0,min(3,w/2)) if r <=c and (h/2+r,h/2+c) in legal_moves]

    def check_timeout(self):
        """ raise exception if timedout, else no-op
        Parameters: None
        -----------

        Returns: None
        -------
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if game.get_player_location == None:
            legal_moves = self.get_init_move(game, legal_moves)

        if len(legal_moves) == 0:
            return (-1,-1)

        best_move = (-1,-1)
        best_score = float("-inf")
        current_depth = 1;
        while True:
            for move in legal_moves:
                try:
                    # The search method call (alpha beta or minimax) should happen in
                    # here in order to avoid timeout. The try/except block will
                    # automatically catch the exception raised by the search method
                    # when the timer gets close to expiring
                    self.check_timeout()
                    nextgame = game.forecast_move(move)
                    if self.method == "minimax":
                        score, _ = self.minimax(nextgame, current_depth-1, False)
                    else:
                        score, _ = self.alphabeta(nextgame, current_depth-1, maximizing_player=False)
                    if score > best_score:
                        best_score, best_move = score, move
                except Timeout:
                    return best_move
            if self.iterative:
                # increase depth by 1 and redo search
                current_depth += 1
                continue
            # iterative search is off and we are done
            break

        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        #initialize vars

        best_move = (-1,-1)
        if maximizing_player:
            player = self
            best_score = float("-inf")
        else:
            player = game.get_opponent(self)
            best_score = float("inf")

        if depth <= 0:
            return float(self.score(game,self)), (-1,-1)

        legal_moves = game.get_legal_moves(player)
        for move in legal_moves:
            self.check_timeout()
            nextgame = game.forecast_move(move)
            newscore, _  = self.minimax(nextgame, depth-1, not maximizing_player)
            if (maximizing_player and newscore > best_score) or \
               (not maximizing_player and newscore < best_score):
                    best_score, best_move = newscore, move

        return best_score, best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        self.check_timeout()

        # check to terminate search
        # we have traversed the max search depth or the game is over
        if maximizing_player:
            player = self
            curr_max = float("-inf")
        else:
            player = game.get_opponent(self)
            curr_min = float("inf")

        if depth <= 0:
            return float(self.score(game,self)), (-1,-1)

        best_move = (-1,-1)
        legal_moves = game.get_legal_moves(player)
        for move in legal_moves:
            self.check_timeout()
            nextgame = game.forecast_move(move)
            newscore, _ = self.alphabeta(nextgame, depth-1, alpha, beta, not maximizing_player)
            if maximizing_player:
                curr_max = max(curr_max, newscore)
                if curr_max >= beta:
                    return curr_max, move
                if (curr_max > alpha):
                    best_move = move
                alpha = max(alpha,curr_max)
            else:
                curr_min = min(curr_min,newscore)
                if curr_min <= alpha:
                    return curr_min, move
                if curr_min < beta:
                    best_move = move
                beta = min(beta,curr_min)

        if maximizing_player:
            return curr_max, best_move
        else:
            return curr_min, best_move

