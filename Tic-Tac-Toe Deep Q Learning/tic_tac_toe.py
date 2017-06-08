import copy
import random

# Red := -1
# Blue := 1
# Empty space := 0

moves = {
    1: (0,0), 2: (0,1), 3: (0,2), 4: (0,3), 5: (0,4), 6: (0,5),
    7: (1,0), 8: (1,1), 9: (1,2), 10: (1,3), 11: (1,4), 12: (1,5),
    13: (2,0), 14: (2,1), 15: (2,2), 16: (2,3), 17: (2,4), 18: (2,5),
    19: (3,0), 20: (3,1), 21: (3,2), 22: (3,3), 23: (3,4), 24: (3,5),
    25: (4,0), 26: (4,1), 27: (4,2), 28: (4,3), 29: (4,4), 30: (4,5),
    31: (5,0), 32: (5,1), 33: (5,2), 34: (5,3), 35: (5,4), 36: (5,5),
}

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement
    legal_moves, make_move, utility, and terminal_test. You may
    override display and successors or you can inherit their default
    methods. You will also need to set the .initial attribute to the
    initial state; this can be done in the constructor."""

    def legal_moves(self, state):
        "Return a list of the allowable moves at this point."
        abstract

    def make_move(self, move, state):
        "Return the state that results from making a move from a state."
        abstract

    def utility(self, state, player):
        "Return the value of this final state to player."
        abstract

    def terminal_test(self, state):
        "Return True if this is a final state for the game."
        return not self.legal_moves(state)

    def to_move(self, state):
        "Return the player whose move it is in this state."
        return state.to_move

    def display(self, state):
        "Print or otherwise display the state."
        print state

    def successors(self, state):
        "Return a list of legal (move, state) pairs."
        return [(move, self.make_move(move, state))
                for move in self.legal_moves(state)]

    def __repr__(self):
        return '<%s>' % self.__class__.__name__



#Class for tic-tac-toe game
class ofttt():
    
    def __init__(self):
        self.board = [[] for i in xrange(6)]
        for i in xrange(6):
            self.board[i] = [0 for j in xrange(6)]
        self.turn = 1
        #Initial state, turn=0 for blue player and turn=1 for red player
        #Board is represented with a matrix None=empty, R=red and B=blue
        self.initial = (self.turn, self.board)  
        
    def legal_moves(self, state):
        "Return a list of the allowable moves at this point."
        legalMoves = []
        board = state[1]
        for i in xrange(6):
            for j in xrange(6):
                if board[i][j] != 1 and board[i][j] != -1:
                    legalMoves.append((i,j))
        #Return the moves that are empty
        return legalMoves

    def make_move(self, move, state):
        "Return the state that results from making a move from a state."
        next_board = copy.deepcopy(state[1])
        x,y = move
        to_move = state[0]
        if to_move == 1:
            next_board[x][y] = 1
        elif to_move == 1:
            next_board[x][y] = -1
        #Return the complement of to_move which is the adversarial turn and the board with the corresponding move
        return (to_move*(-1), next_board)
    
    
    def utility(self, state, player):
        "Return the value of this final state to player."
        #First, verify wether the board is full or not
        if len(self.legal_moves(state)) == 0:
            victory = False
            board = state[1]
            #Now, we look for draw or victory using the test win function
            if player == 1:
                for i in xrange(6):
                    for j in xrange(6):
                        if board[i][j] == 1 and not victory:
                            victory = test_win(board,(i,j),1)
            else:
                for i in xrange(6):
                    for j in xrange(6):
                        if board[i][j] == -1 and not victory:
                            victory = test_win(board,(i,j),-1)
            #Draw Case
            if not victory:
                return 0
        #Lose case
        if state[0] == player:
            return -10
        #Victory case
        else:
            return 10
        
    def reward(self, state, move, player):
        #First, verify wether the board is full or not
        victory = False
        board = state[1]
        (x,y) = move
        board[x][y] = player
        if len(self.legal_moves(state)) == 0:
            #Now, we look for draw or victory using the test win function
            if player == 1:
                for i in xrange(6):
                    for j in xrange(6):
                        if board[i][j] == 1 and not victory:
                            victory = test_win(board,(i,j),1)
            else:
                for i in xrange(6):
                    for j in xrange(6):
                        if board[i][j] == -1 and not victory:
                            victory = test_win(board,(i,j),-1)
            #Draw Case
            if not victory:
                return 0
        if player == 1:
            for i in xrange(6):
                for j in xrange(6):
                    if board[i][j] == 1 and not victory:
                        victory = test_win(board,(i,j),1)
        else:
            for i in xrange(6):
                for j in xrange(6):
                    if board[i][j] == -1 and not victory:
                        victory = test_win(board,(i,j),-1)
        if not victory:
            return 0
        
        if player == 1:
            return 1
        else:
            return -1

        
    def terminal_test(self, state):
        "Return True if this is a final state for the game."
        board = state[1]
        victory = False
        for i in xrange(6):
            for j in xrange(6):
                if board[i][j] == -1 and not victory:
                    victory = test_win(board,(i,j),-1)
                elif board[i][j] == 1 and not victory:
                    victory = test_win(board,(i,j),1)
        #Return true if there is a victory or the board is full.
        return len(self.legal_moves(state)) == 0 or victory

    def to_move(self, state):
        "Return the player whose move it is in this state."
        return state[0]

    def display(self, state):
        "Print or otherwise display the state."
        for i in state[1]:
            print i

    def successors(self, state):
        "Return a list of legal (move, state) pairs."
        return [(move, self.make_move(move, state))
                for move in self.legal_moves(state)]

    def next_state(self, a_t, state):
        n_state = self.make_move(moves[a_t],state)
        r_t = self.reward(state, moves[a_t], 1)
        terminal = self.terminal_test(n_state)
        return n_state, r_t, terminal

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def alphabeta_search(state, game, d=float('inf'), cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state, player)
        v = -float('inf')
        for (a, s) in game.successors(state):
            v = max(v, min_value(s, alpha, beta, depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state, player)
        v = float('inf')
        for (a, s) in game.successors(state):
            v = min(v, max_value(s, alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or
                   (lambda state,depth: depth>d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state, player: game.utility(state, player))
    action, state = argmax(game.successors(state),
                           lambda ((a, s)): min_value(s, -float('inf'), float('inf'), 0))
    return action

#Function for minimax player
def fn(state,player):
    maxRed = 1
    maxBlue = 1        #Counters for maximum number of pieces
    board = state[1]
    for i in xrange(6):
        for j in xrange(6):
            if board[i][j] != None:         #Looking for some color
                x,y = i,j 
                color = board[i][j] 
                count = 1
                auxcount = 1
                #Count for diagonal right upside down
                while y+1 < len(board) and x-1 >= 0 and board[x-1][y+1] == color and count != 4:
                    x -= 1
                    y += 1    
                    count += 1          
                auxcount = max(auxcount, count)
                x,y = i,j
                count = 1
                #Count for same column
                while x+1 < len(board) and board[x+1][y] == color and count != 4:
                    x += 1
                    count += 1
                auxcount = max(auxcount, count)
                x,y = i,j
                count = 1
                #Count for same row
                while y+1 < len(board) and board[x][y+1] == color and count != 4:
                    y += 1
                    count += 1
                auxcount = max(auxcount, count)
                x,y = i,j
                count = 1
                #Count for diagonal right upside
                while x+1 < len(board) and y+1 < len(board) and board[x+1][y+1] == color and count != 4:
                    x += 1
                    y += 1
                    count += 1
                auxcount = max(auxcount, count)
                #Assign current color to the corresponding counter
                if color == 'R':
                    maxRed = max(maxRed, auxcount)
                else:
                    maxBlue = max(maxBlue, auxcount)
    #Verify the current player and assign the evaluation function
    if player == 1:
        if maxRed == 4:
            return 10
        if maxBlue == 4:
            return -10
        return maxRed-maxBlue
    else:
        if maxBlue == 4:
            return 10
        if maxRed == 4:
            return -10
        return maxBlue-maxRed


def test_win(board, pos, color):
    x,y = pos
    count = 1
    while y+1 < len(board) and x-1 >= 0 and board[x-1][y+1] == color and count != 4:
        x -= 1
        y += 1
        count += 1
    if count == 4:
        return True
    x,y = pos
    count = 1
    while x+1 < len(board) and board[x+1][y] == color and count != 4:
        x += 1
        count += 1
    if count == 4:
        return True
    x,y = pos
    count = 1
    while y+1 < len(board) and board[x][y+1] == color and count != 4:
        y += 1
        count += 1
    if count == 4:
        return True
    x,y = pos
    count = 1
    while x+1 < len(board) and y+1 < len(board) and board[x+1][y+1] == color and count != 4:
        x += 1
        y += 1
        count += 1
    if count == 4:
        return True
    return False

def query_player(game, state):
    "Make a move by querying standard input."
    x,y = map(int,raw_input('Your move? ').split(" "))
    while (x,y) not in game.legal_moves(state):         #Verify wether it is a legal move or not
        x,y = map(int,raw_input('Your move? ').split(" "))
    return x,y

def smart_player(game, state):
    return alphabeta_search(state, game, d = 2, eval_fn = fn)

def random_player(game, state):
    index = len(game.legal_moves)
    play = random.randrange(0,index)
    return moves[play]

def play_game(game, *players):
    "Play an n-person, move-alternating game."
    state = game.initial
    for item in state[1]:
        for el in item:
            if el == None:
                print '_',
            elif el == 'R':
                print 'X',
            else:
                print 'O',
        print
    print
    pl = state[0]
    board = state[1]
    while True:
        for player in players:
            move = player(game, state)
            state = game.make_move(move, state)
            for item in state[1]:
                for el in item:
                    if el == None:
                        print '_',
                    elif el == 'R':
                        print 'X',
                    else:
                        print 'O',
                print
            print
            if game.terminal_test(state):
                w = game.utility(state, 0)
                if w == -10:
                    return 'RED WINS'
                elif w == 10:
                    return 'BLUE WINS'
                else:
                    return 'DRAW'
                    
