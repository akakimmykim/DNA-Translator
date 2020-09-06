# -*- coding: utf-8 -*-

import numpy as np

np.random.seed(1)

def create_board():
    return np.zeros((3,3), dtype=int)

def place(board, player, position):
    if (board[position[0],position[1]] == 0):
        board[position[0],position[1]] = player
        print(board)
        return True
    else:
        #print("Already Taken")
        return False

def checkWin(board, player, position):
    size = 3
    checkBoard = np.full(size, player, dtype=int)
    # check horizontal
    if (np.all(board[:,position[1]] == checkBoard)):
        return True
    # check vertical
    if (np.all(board[position[0], :] == checkBoard)):
        return True
    # check diagonal
    if (np.all(np.diagonal(board) == checkBoard) or np.all(np.diagonal(np.transpose(board)))):
        return True
    else:
        return False

def getPos ():
    # For Humans
    ''' 
    print("Please type the row (1-3) and column (1-3) with a space in between... ex 2 2 is the center")
    choice = input("Position: ")
    choice_list = choice.split()
    if (len(choice_list) == 2):
        return ((int(choice_list[0])-1, int(choice_list[1])-1))
    else: 
        print ("Not a Valid Position")
        return ((-1,-1))
    '''
    # For Bots
    return ((np.random.randint(3), np.random.randint(3)))

def playGame(board):
    countTurn = 0
    for count in range(9):
        Player = 1
        checkPlace = False
        winner = 0
        for turn in range (2):
            position = (-1,-1)
            while(True):
                #print("Player ", Player, "'s Turn: Please type 2 integers ")
                position = getPos()
                checkPlace = place(board1, Player, position)
                if(position != (-1,-1) and checkPlace == True):
                   break   
            if (checkWin(board1, Player, position)):
                print("Player", Player, "Wins!")
                winner = Player
                break
            Player += 1
        if (winner != 0):
            board.fill(0)
            return winner
    print ("TIE... GG")
    board.fill(0)
    return 0

def playStratGame(board):
    for count in range(9):
        Player = 1
        checkPlace = False
        winner = 0
        for turn in range (2):
            if (count == 0 and Player == 1):
                position = (1,1)
                place(board1, Player, position)
                Player += 1
            else:
                position = (-1,-1)
                while(True):
                    #print("Player ", Player, "'s Turn: Please type 2 integers ")
                    position = getPos()
                    checkPlace = place(board1, Player, position)
                    if(position != (-1,-1) and checkPlace == True):
                       break   
                if (checkWin(board1, Player, position)):
                    print("Player", Player, "Wins!")
                    winner = Player
                    break
                Player += 1
        if (winner != 0):
            board.fill(0)
            return winner
    print ("TIE... GG")
    board.fill(0)
    return 0
        
# MAIN
board1 = create_board()
gameNum = 3
player_1_Wins = 0
player_2_Wins = 0
for gameCount in range (gameNum):
    wonner = playStratGame(board1)
    if (wonner == 1):
        player_1_Wins += 1
    elif (wonner == 2):
        player_2_Wins +=1
        
print("Player 1 Won", player_1_Wins, "times")
print("Player 2 Won", player_2_Wins, "times")