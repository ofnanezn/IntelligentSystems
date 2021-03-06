# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def general_ui_search(problem, frontier):
    visited = {}
    state = problem.getStartState()
    frontier.push((state, []))
    visited[state] = 'gray'
    while not frontier.isEmpty():
        u, actions = frontier.pop()
        if problem.isGoalState(u):
            return actions
        for v, action, cost in problem.getSuccessors(u):
            if not v in visited:
                visited[v] = 'gray'
                frontier.push((v, actions + [action]))
        visited[u] = 'black'
    return []

"""def general_ui_search(problem, frontier, isPill,dots):
    visited = {}
    state = problem.getStartState()
    frontier.push((state, []))
    visited[state] = 'gray'
    while not frontier.isEmpty():
        u, actions = frontier.pop()
        if problem.isGoalState(u):
            return u,len(actions)
        for v, action, cost in problem.getSuccessors2(u,isPill,dots):
            if not v in visited:
                visited[v] = 'gray'
                frontier.push((v, actions + [action]))
        visited[u] = 'black'
    return state,[]"""

"""def general_ui_search(problem, frontier, isPill, pills, dots):
    visited = {}
    state = problem.getStartState()
    frontier.push(state)
    visited[state] = 'gray'
    d = {state: 0}
    dist = {}
    while not frontier.isEmpty():
        u = frontier.pop()
        for v, action, cost in problem.getSuccessors2(u,isPill,dots):
            if not v in d:
                d[v] = d[u] + 1
                frontier.push(v)
            if isPill and v in pills:
                dist[v] = d[v]
            if not isPill and v in dots:
                dist[v] = d[v]
    return dist"""

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return general_ui_search(problem, util.Queue())


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def general_search(problem, frontier):
    visited = {}
    state = problem.getStartState()
    frontier.push((state, [], 0))
    while not frontier.isEmpty():
        u, actions, path_cost = frontier.pop()
        if problem.isGoalState(u):
            return  actions
        if not u in visited:
            for v, action, cost in problem.getSuccessors(u):
                frontier.push((v, actions + [action], path_cost + cost))
        visited[u] = 'black'
    return []

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    s = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((s,[]),0)
    depth = {}
    depth[s] = 0
    while not frontier.isEmpty():
        state,path = frontier.pop()
        if problem.isGoalState(state):
            #print path
            return path
        for next_state,direction,new_cost in problem.getSuccessors(state):
            if next_state not in depth:
                depth[next_state] = depth[state] + new_cost
                frontier.push((next_state,path+[direction]),depth[next_state]+heuristic(next_state,problem))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
