import numpy as np
from copy import deepcopy
import sys


def is_coordinate_available(x, y, game_state):
    global K
    if x < 1 or x > K or y < 1 or y > K or game_state[x][y] == 1 or game_state[x][y] == -2 or game_state[x][y] == -1:
        return False
    return True


def get_valid_coordinates(coordinates, game_state):
    valid_coordinates = []
    neighs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in neighs:
        x = coordinates[0] + i[0]
        y = coordinates[1] + i[1]
        if is_coordinate_available(x, y, game_state):
            valid_coordinates.append([x, y])
    return valid_coordinates


def get_bot_position(game_state):
    global K
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if game_state[i][j] == -2:
                return [i, j]


def get_player_position(game_state):
    global K
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if game_state[i][j] == -1:
                return [i, j]


def is_game_over(game_state):
    position_b = get_bot_position(game_state)
    position_p = get_player_position(game_state)
    if len(get_valid_coordinates(position_p, game_state)) == 0:
        return +1
    if len(get_valid_coordinates(position_b, game_state)) == 0:
        return -1
    return 0


def get_possible_states(game_state, who):
    pos_b = get_bot_position(game_state)
    pos_p = get_player_position(game_state)
    states = []
    if who == -1:
        coordinates = pos_p
        opponent_coordinates = pos_b
    else:
        coordinates = pos_b
        opponent_coordinates = pos_p

    for position in get_valid_coordinates(coordinates, game_state):
        temporary = deepcopy(game_state)
        temporary[coordinates[0]][coordinates[1]] = 0
        temporary[position[0]][position[1]] = who
        for lock in get_valid_coordinates(opponent_coordinates, temporary):
            temporary2 = deepcopy(temporary)
            temporary2[lock[0]][lock[1]] = 1
            states.append(temporary2)
    return states


def minmax3x3(game_state, is_max):
    status = is_game_over(game_state)
    if status != 0:
        return status

    if is_max:
        max_val = -np.inf
        for state in get_possible_states(game_state, -1):
            max_val = max(minmax3x3(state, not is_max), max_val)
        return max_val
    else:
        min_val = np.inf
        for state in get_possible_states(game_state, -2):
            min_val = min(minmax3x3(state, not is_max), min_val)
        return min_val


def heuristic_evaluation(game_state):
    [bx, by] = get_bot_position(game_state)
    [px, py] = get_player_position(game_state)
    nr_bot = len(get_valid_coordinates([bx, by], game_state))
    nr_p = len(get_valid_coordinates([px, py], game_state))
    return nr_bot - nr_p


def minmax5x5(game_state, is_max, depth, a, b):
    status = heuristic_evaluation(game_state)
    if depth == 0:
        return status
    depth = depth - 1
    if is_max:
        max_val = -np.inf
        for state in get_possible_states(game_state, -1):
            val = minmax5x5(state, not is_max, depth, a, b)
            max_val = max(val, max_val)
            a = max(max_val, a)
            if a >= b:
                return max_val
        return max_val
    else:
        min_val = np.inf
        for state in get_possible_states(game_state, -1):
            val = minmax5x5(state, not is_max, depth, a, b)
            min_val = min(val, min_val)
            a = min(min_val, a)
            if a >= b:
                return min_val
        return min_val


def initialize_board():
    global K
    if K == 3:
        pos_b = [1, 2]
        pos_p = [3, 2]
        table = np.ones((K + 2, K + 2))
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                table[i][j] = 0
        table[pos_p[0], pos_p[1]] = -1
        table[pos_b[0], pos_b[1]] = -2
        return table

    if K == 5:
        pos_b = [1, 3]
        pos_p = [5, 3]
        table = np.ones((K + 2, K + 2))
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                table[i][j] = 0
        table[pos_p[0], pos_p[1]] = -1
        table[pos_b[0], pos_b[1]] = -2
        return table


def get_player_input(game_state):
    print_game_state(game_state)

    print("Player lép:")
    pcoord = get_player_position(game_state)
    inx = int(input("X coordinate for the move: "))
    iny = int(input("Y coordinate for the move: "))

    while (not is_coordinate_available(inx, iny, game_state) or (pcoord[0] == inx and pcoord[1] == iny) or
           inx > K or inx < 1 or iny > K or inx < 1
           or not abs(pcoord[0] - inx) <= 1 or not abs(pcoord[1] - iny) <= 1):
        inx = int(input("X coordinate for the move: "))
        iny = int(input("Y coordinate for the move: "))
    game_state[pcoord[0], pcoord[1]] = 0
    game_state[inx, iny] = -1
    print_game_state(game_state)

    inx = int(input("X coordinate for the block: "))
    iny = int(input("Y coordinate for the block: "))

    while not is_coordinate_available(inx, iny, game_state):
        inx = int(input("X coordinate for the block: "))
        iny = int(input("Y coordinate for the block: "))
    game_state[inx][iny] = 1
    print_game_state(game_state)


def get_best_move(game_state, player, depth):
    if K == 3:
        best_move = None
        moves = get_possible_states(game_state, player)
        max_eval = -np.inf
        for move in moves:
            evaluation = minmax3x3(game_state, True)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
        return best_move
    elif K == 5:
        best_move = None
        moves = get_possible_states(game_state, player)
        a = -np.inf
        b = np.inf
        max_eval = -np.inf
        for move in moves:
            evaluation = minmax5x5(game_state, True, depth, a, b)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
        return best_move


def print_game_state(game_state):
    global K
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if game_state[i][j] == 0:
                print("[ ]", end="")
            elif game_state[i][j] == 1:
                print("[X]", end="")
            elif game_state[i][j] == -1:
                print("[P]", end="")
            elif game_state[i][j] == -2:
                print("[B]", end="")

        print()
    print()
    print()
    sys.stdout.flush()


def player_first(game_state):
    ind = 0
    depth = 2
    while True:
        ind += 1
        x, y = get_player_position(game_state)
        if len(get_valid_coordinates([x, y], game_state)) == 0:
            print_game_state(game_state)
            print("AI won !")
            break

        get_player_input(game_state)

        if ind % 4 == 0:
            depth += 1

        print("Player moved: \n\n")
        game_state = get_best_move(game_state, -2, depth)

        if game_state is None:
            print("Player won ! ")
            break


def bot_first(game_state):
    ind = 0
    depth = 2

    while True:
        ind += 1

        if ind % 3 == 0:
            depth += 1

        print("AI moved: \n\n")
        game_state = get_best_move(game_state, -2, depth)

        if game_state is None:
            print("Player won ! ")
            break

        x, y = get_player_position(game_state)
        if len(get_valid_coordinates([x, y], game_state)) == 0:
            print_game_state(game_state)
            print("AI won !")
            break

        get_player_input(game_state)


def start_game(starting_entity):
    board = initialize_board()
    if starting_entity == -1:
        player_first(board)
    elif starting_entity == -2:
        bot_first(board)


if __name__ == '__main__':
    # -1 -> PLAYER
    # -2 -> BOT
    who_starts = -2
    K = 5
    start_game(who_starts)
