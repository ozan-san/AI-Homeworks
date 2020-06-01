#!/usr/bin/python3
'''
Assignment 1: Search (A* Algorithm)
@author: Ozan Şan, Zdeněk Rozsypálek, and the KUI-2019 team
@contact: sanozan@fel.cvut.cz
'''

import time
import kuimaze
import os



from heuristics import heur

class Agent(kuimaze.BaseAgent):
    '''
    Simple example of agent class that inherits kuimaze.BaseAgent class 
    '''
    def __init__(self, environment):
        self.environment = environment
        
    def find_path(self):
        '''
        A* Pathfinding Algorithm implementation.
        
        Args:
            None
        
        Returns:
            list: Shortest path in the maze, as a list of tuples
                in the form [(x1, y1), (x2, y2)...].
        Expects to return a path_section as a list of positions [(x1, y1), (x2, y2), ... ].
        '''
        observation = self.environment.reset() 
        goal = observation[1][0:2]
        start = observation[0][0:2]                               # initial state (x, y)
        
        parents = {} # For fast access, we keep the parents in a dictionary
        parents[start] = None # Starting node has no parent
        openList = [] # OpenList is just a list containing nodes and costs
        openList.append((start, heur(start, goal), 0, heur(start,goal)))
        # The structure of the elements in openList:
        # ((x, y), f, g, h), where (x,y) is the state, and f,g,h are costs, respectively.
        # Note: This is implementable with PriorityQueues as well,
        # But this would complicate the situation far worse,
        # Especially when we need to update the costs for entries inside openList.
        # This is why the openList is just a plain python list.
        # This hurts the performance, but this is the best I could do.
        closedList = set([])
        # ClosedList is a Hash Table containing states that are done with.
        # The entries in ClosedList are only positions (x,y) of the states.
        found_path = False
        # This variable is useful when we break our loop, and reconstruct the path.
        while openList: # While openList is not empty:
            # ... Find the node with the lowest f cost.
            current = openList[0] 
            for item in openList:
                if item[1] <= current[1]:
                    if item[3] < current[3]:
                        current = item
            # ... Find the node with the lowest f cost.
            # If the f costs are the same (lowest), break the tie with h cost.
            openList.remove(current)
            closedList.add(current[0])
            # The selected node is done with. We can remove it, and move on.
            if current[0] == goal:
                # If the current node to be processed is the goal, we are done!
                # And, we need to reconstruct the path after this.
                found_path = True
                break
            
            children = self.environment.expand(current[0])
            # Explored children of the current node.
            for child in children:
                # Structure of a child: ((x,y), moveCost)
                if child[0] in closedList: # If we are already done with this child,
                    continue # Move on to the next child.
                
                newCost = current[2] + child[1]
                # This is the candidate new G cost for the children.
                # g_cost(start, current) + moveCost
                
                openList_contains_child = False
                # If the openList does not contain this child, we should add it.
                
                cost_beaten = False
                index_of_beaten = -1
                # If we are already going to process this child,
                # We may need to update its costs.
                for i in range(len(openList)): 
                    if openList[i][0] == child[0]:
                        openList_contains_child = True #It's in the list!
                        if openList[i][2] > newCost: #Do we beat the cost?
                            cost_beaten = True # We do!
                            index_of_beaten = i # And it's at this index, in which we beat the cost
                            break # 
                
                if (cost_beaten or not openList_contains_child):
                    pos = child[0]
                    g = newCost # candidate g cost.
                    h = heur(child[0], goal) # O(1) operation. Cheap.
                    f = g + h
                    parents[child[0]] = current[0] # update or reset the parent.
                    # update (or set) the parent, since we've found a shorter path
                    
                    if cost_beaten: # This means it should be in the open list.
                        openList[index_of_beaten] = (pos, f, g, h)
                        # construct a new entry for openList
                        # in the form ((x,y), f, g, h)
                    elif not openList_contains_child:
                        # We should add the new child to the openList.
                        openList.append((pos, f, g, h))
            #self.environment.render()
            #time.sleep(0.05)
        if found_path:
            path = []
            node = goal
            # Starting from the goal, backtracking our steps.
            while node:
                path.append(node)
                # [goal, previous1, prev2...., start]
                node = parents[node]
                # Check out the parent.
                # print(node)
            return path[::-1] # Reverse the path, and return.
        return None # No path found.
                    
        
            
if __name__ == '__main__':

    MAP = 'maps/normal/normal11.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 

    path = agent.find_path()
    print(path)
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(3)
