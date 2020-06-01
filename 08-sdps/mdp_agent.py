from copy import deepcopy
from random import choice


def init_policy(problem):
    policy = dict()
    for state in problem.get_all_states():
        if problem.is_goal_state(state):
            policy[state.x, state.y] = None
            continue
        actions = [action for action in problem.get_actions(state)]
        policy[state.x, state.y] = choice(actions)
    return policy


def init_utils(problem):
    '''
    Initialize all state utilities to zero except the goal states
    :param problem: problem - object, for us it will be kuimaze.Maze object
    :return: dictionary of utilities, indexed by state coordinates
    '''
    utils = dict()
    x_dims = problem.observation_space.spaces[0].n
    y_dims = problem.observation_space.spaces[1].n

    for x in range(x_dims):
        for y in range(y_dims):
            utils[(x, y)] = 0

    for state in problem.get_all_states():
        utils[(state.x, state.y)] = state.reward  # problem.get_state_reward(state)
    return utils


def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    """
    Find a suitable policy for the agent, using value iteration method.
    :param problem: object, of type kuimaze.Maze
    :param discount_factor: float
    :param epsilon: float
    :return: dictionary of actions, indexed by state coordinate pairs
    """

    # We initialize V_0(s) to be 0, except for the terminal states.
    utilities = init_utils(problem)
    # V_k+1(s) is stored in new_utilities, and assigned into utilities in each iteration.
    new_utilities = init_utils(problem)
    # We record the states for ease of use, and to refrain from asking the object to retrieve
    # all the states at each iteration, for they do not change anyway.
    states = problem.get_all_states()
    # The policy is updated at each iteration. This presents no extra computation, since
    # we already compute a max, we can also record the action that results in max utility.
    policy_found = init_policy(problem)
    # To determine convergence, we initialize two different variables, a boolean called
    # non_convergence, and the stop factor. If no value change of magnitude stop_factor
    # occurs in an iteration, we do not set the non_convergence flag, resulting in the termination
    # of the main loop.
    non_convergence = True
    stop_factor = epsilon * (1 - discount_factor) / discount_factor

    while non_convergence:
        # We first assume the values to be converged. If this is proven otherwise,
        # we will do another pass of Bellman updates.
        non_convergence = False
        # For each state s in S...
        for state in states:
            # If the state is terminal, we do not need to change anything.
            if problem.is_terminal_state(state):
                # We move on to the next state.
                continue

            # We retrieve all actions a in A for the state in question.
            actions = problem.get_actions(state)
            # We initialize the maximum utility found for the state to be -inf, to be replaced later.
            val_max = -1 * float('inf')
            # This is crucial: We also record the action resulting in the maximum value for V_k+1(s).
            # This is the step that determines the policy, which is the whole point of the exercise :)
            action_max = None

            for action in actions:
                # We retrieve all outcomes and probabilities of the action we took.
                possibilities = problem.get_next_states_and_probs(state, action)
                # There needs to be a (weighted) summation of the values for V_k(s').
                # This is where it will be accumulated.
                value_of_action = 0
                for possibility in possibilities:
                    # We simply unpack the state and probabilities first.
                    next_state = possibility[0]
                    probability = possibility[1]
                    # The value is incremented by p(s'|a,s)*V_k(s')
                    value_of_action += probability * utilities[(next_state.x, next_state.y)]
                # If we have found a better outcome from the selected action, we update the variables.
                if value_of_action > val_max:
                    action_max = action
                    val_max = value_of_action
            # Value is first discounted.
            val_max *= discount_factor
            # Then, R(s) is added.
            val_max += problem.get_state_reward(state)
            # We now have computed R(s) + (disc)*max_(a) sum_(s') p(s'|s, a) V_k(s'), which is V_k+1(s).
            # This is stored in the new utilities.
            new_utilities[(state.x, state.y)] = val_max
            # And since we have found the optimal action as well, we can store it in policy dictionary as well.
            policy_found[state.x, state.y] = action_max

            # We see if the resulting change is larger than the stop factor. If so, we will do another iteration.
            if abs(new_utilities[(state.x, state.y)] - utilities[(state.x, state.y)]) > stop_factor:
                non_convergence = True
            # We update the utilities.
            utilities[(state.x, state.y)] = new_utilities[(state.x, state.y)]
    # Now, we can return the policy.
    return policy_found

def policy_evaluation(problem, policy, discount_factor, max_iterations = 50, epsilon = 0.01):
    """
    Policy Evaluation: Evaluates policies in a bottom up dynamic programming fashion.
    :param problem: object, of type kuimaze.MDPMaze
    :param policy: dict, with state-action pairs
    :param discount_factor: float
    :param max_iterations: int, the limit of dynamic programming iterations
    :param epsilon: float, factor in error computation
    """
    # We first initialize utility dictionaries. These contain the living rewards for non-terminal
    # states, and rewards for terminal states by default.
    utility = init_utils(problem)
    new_utility = init_utils(problem)
    # We retrieve all states for ease of use.
    states = problem.get_all_states()
    # The current iteration count is recorded, and incremented each iteration.
    iteration_count = 0
    # We also have a boolean to indicate whether we have arrived at a stable solution or not.
    divergence = True
    # This is the same as the case in value iteration.
    stop_factor = epsilon * (1 - discount_factor) / discount_factor

    while iteration_count < max_iterations and divergence:
        # We set the loop control variables.
        iteration_count += 1
        divergence = False

        for state in states:
            # If the state in question is terminal, we don't need to change anything.
            if problem.is_terminal_state(state):
                continue
            # The possible outcomes of the action determined by the policy is retrieved.
            possibilities = problem.get_next_states_and_probs(state, policy[(state.x, state.y)])
            # The sum of the values for possible outcomes will be held in this variable.
            sum = 0
            for possibility in possibilities:
                # We unpack the possibility into coordinate and probability.
                state_coordinates = possibility[0]
                probability = possibility[1]
                # Incrementing sum by p(s'|s, a)* V_k Pi (s')
                sum += probability * utility[(state_coordinates.x, state_coordinates.y)]
            # The sum is discounted, and stored in new_utility.
            sum *= discount_factor
            new_utility[(state.x, state.y)] = sum
            # If we have a difference larger than stop_factor, we must continue.
            if abs(sum - utility[(state.x, state.y)]) > stop_factor:
                divergence = True
        # The utility is updated. Shallow copy won't be useful here, so we deepcopy it.
        utility = deepcopy(new_utility)
    return utility


def find_policy_via_policy_iteration(problem, discount_factor):
    """
    Find Policy via Policy Iteration.
    :param problem: object, of type kuimaze.MDPMaze
    :param discount_factor: float
    """
    # We start by initializing two dictionaries for policies.
    policy = init_policy(problem)
    new_policy = init_policy(problem)
    # As usual, retrieving the states for ease of use.
    states = problem.get_all_states()
    # We keep record of the stability of the policy. If there is no change
    # between subsequent iterations, this value will be false, resulting
    # in the termination of the main loop.
    policy_unstable = True
    while policy_unstable:
        # We set this value to False, it will be set back to true
        # when we detect a need to change the policy.
        policy_unstable = False
        # We evaluate the current policy.
        evaluation = policy_evaluation(problem, policy, discount_factor)
        # We iterate over all states, looking for ways to improve.
        for state in states:
            # If the state in question is terminal, there's nothing to do.
            if problem.is_terminal_state(state):
                continue
            else:
                # We retrieve all the possible actions for the state in question.
                # We will select the best among them for our new policy.
                actions = problem.get_actions(state)
                # For argmax operation, we initialize two values.
                max_value = -float('inf')
                max_action = None

                for action in actions:
                    # The possible outcomes of the action-state pair in question is retrieved.
                    possibilities = problem.get_next_states_and_probs(state, action)
                    # The sum will be accumulated in this variable.
                    sum = 0
                    for possibility in possibilities:
                        # Unpacking the possibility.
                        probability = possibility[1]
                        next_state = possibility[0]
                        # Incrementation by p(s' | s, a) * V^(pi(i))(s')
                        # We omit the discount, because in argmax, it does not matter.
                        sum += probability * evaluation[(next_state.x, next_state.y)]
                    # If we have a better action, we update the variables accordingly.
                    if sum > max_value:
                        max_value = sum
                        max_action = action
                # The new policy for the state in question is determined.
                new_policy[(state.x, state.y)] = max_action
                # If we have a change in policy, we must run another loop.
                if policy[(state.x, state.y)] != new_policy[(state.x, state.y)]:
                    policy_unstable = True
        # We assign the new policy to the old one, resulting in continuous improvement.
        policy = deepcopy(new_policy)
    return policy

