import math

# Initialize the board
board = [['-' for _ in range(3)] for _ in range(3)]

# Check for winner or draw
def check_winner(board):
    # Check rows, columns, and diagonals
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != '-':
            return row[0]
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != '-':
            return board[0][col]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '-':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '-':
        return board[0][2]
    # Check for draw
    if all(cell != '-' for row in board for cell in row):
        return 'Draw'
    return None

# Minimax function
def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'O': return 10 - depth
    if winner == 'X': return depth - 10
    if winner == 'Draw': return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = '-'
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = '-'
                    best_score = min(score, best_score)
        return best_score

# Find best move for AI
def best_move(board):
    best_score = -math.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == '-':
                board[i][j] = 'O'
                score = minimax(board, 0, False)
                board[i][j] = '-'
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

# Main game loop (player vs. AI)
def play_game():
    player = 'X'
    ai = 'O'
    while True:
        print_board(board)
        if check_winner(board):
            print("Game Over:", check_winner(board))
            break
        if player == 'X':  # Player's turn
            row, col = map(int, input("Enter row and col: ").split())
            if board[row][col] == '-':
                board[row][col] = player
                player = ai
        else:  # AI's turn
            row, col = best_move(board)
            board[row][col] = ai
            player = 'X'

def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

# Start the game
play_game()
