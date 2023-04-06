# Gridworld

For this assignment you will create a reinforcement learner that uses either SARSA or Q-learning to navigate a gridworld.  Your program will accept several command line arguments, read in a file corresponding to a gridworld, and act in a way to balance exploring and exploiting that world.  

Your assignment will be run by a WPI faculty member who is not the most technically proficient.  Please make it as easy as possible to run your code from the command line.
Input format
Your program will accept several command-line arguments that control its behavior:
The name of the file representing the gridworld
Reward for each action the agent takes.  You may assume this value is non-positive.
Gamma, the discount parameter
How many seconds to run for (can be <1 second)
P(action succeeds):  the transition model. 

The input file represents the gridworld the agent will explore and solve.  
The Gridworld
You will read in a tab-delimited file representing a gridworld.  Each character will be one of the following:
S:  represents where the agent will start
X:  represents an impassable barrier
0:  represents an empty square
[-9,9] (except 0):  represents a terminal square where an agent will receive that numeric reward.  Boards can have multiple positive and multiple negative terminal states.  

Any movement that would take the agent into a barrier or off the grid results in the agent staying in place, but receiving whatever reward it normally would for taking an action.  
