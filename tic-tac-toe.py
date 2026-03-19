# ============================================================
# TASK 2: TIC-TAC-TOE AI (Minimax with Alpha-Beta Pruning)
# CodSoft AI Internship
# ============================================================

import math

# ---------- Board Setup ----------
def create_board():
    return [' '] * 9

def print_board(board):
    print("\n")
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---+---+---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---+---+---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print("\n")

def print_positions():
    print("\nBoard positions (enter number to place your mark):")
    print(" 1 | 2 | 3 ")
    print("---+---+---")
    print(" 4 | 5 | 6 ")
    print("---+---+---")
    print(" 7 | 8 | 9 \n")

# ---------- Game Logic ----------
WIN_COMBOS = [
    (0,1,2), (3,4,5), (6,7,8),  # rows
    (0,3,6), (1,4,7), (2,5,8),  # columns
    (0,4,8), (2,4,6)             # diagonals
]

def check_winner(board, player):
    return any(board[a] == board[b] == board[c] == player for a, b, c in WIN_COMBOS)

def is_draw(board):
    return ' ' not in board

def get_available_moves(board):
    return [i for i, cell in enumerate(board) if cell == ' ']

# ---------- Minimax with Alpha-Beta Pruning ----------
def minimax(board, depth, is_maximizing, alpha, beta):
    if check_winner(board, 'X'):   # AI wins
        return 10 - depth
    if check_winner(board, 'O'):   # Human wins
        return depth - 10
    if is_draw(board):
        return 0

    if is_maximizing:
        best = -math.inf
        for move in get_available_moves(board):
            board[move] = 'X'
            score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = ' '
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break  # Beta cutoff
        return best
    else:
        best = math.inf
        for move in get_available_moves(board):
            board[move] = 'O'
            score = minimax(board, depth + 1, True, alpha, beta)
            board[move] = ' '
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break  # Alpha cutoff
        return best

def best_move(board):
    best_score = -math.inf
    move = None
    for m in get_available_moves(board):
        board[m] = 'X'
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[m] = ' '
        if score > best_score:
            best_score = score
            move = m
    return move

# ---------- Main Game Loop ----------
def play_game():
    print("=" * 50)
    print("    TIC-TAC-TOE AI  -  CodSoft AI Task 2")
    print("=" * 50)
    print("You are 'O'  |  AI is 'X'")
    print_positions()

    board = create_board()

    # Decide who goes first
    first = input("Do you want to go first? (y/n): ").strip().lower()
    human_turn = first == 'y'

    while True:
        print_board(board)

        if human_turn:
            # Human move
            while True:
                try:
                    pos = int(input("Your move (1-9): ")) - 1
                    if pos < 0 or pos > 8:
                        print("Please enter a number between 1 and 9.")
                    elif board[pos] != ' ':
                        print("That cell is already taken! Choose another.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Enter a number between 1 and 9.")
            board[pos] = 'O'

            if check_winner(board, 'O'):
                print_board(board)
                print("🎉 Congratulations! You won! (That's rare!)")
                break
        else:
            # AI move
            print("AI is thinking...")
            move = best_move(board)
            board[move] = 'X'
            print(f"AI placed at position {move + 1}")

            if check_winner(board, 'X'):
                print_board(board)
                print("🤖 AI wins! Better luck next time.")
                break

        if is_draw(board):
            print_board(board)
            print("🤝 It's a draw!")
            break

        human_turn = not human_turn

    play_again = input("\nPlay again? (y/n): ").strip().lower()
    if play_again == 'y':
        play_game()
    else:
        print("Thanks for playing!")


if __name__ == "__main__":
    play_game()