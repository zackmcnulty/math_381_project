
import random as r
import numpy as np
import itertools as itr

from Student import *


def room_assign(room_dim, student_list, periods):
#room_dim = list of room dimensions for each room (e.g. seat length x seat width)
#periods = number of classes student has per day

    #determine number of students (seats) in each room
    room_size = []
    for dim in room_dim:  # for each set of dim in room_dim
        room_size.append(dim[0] * dim[1])  # size = r * c

    #determine total number of students, generate list of students
    num_students = sum(room_size)  # total number of students
    student_numbers = list(range(num_students))  # list of student numbers

    for _ in range(periods): #iterates through number of class periods
        room_list = []
        r.shuffle(student_numbers) #shuffle list of students randomly

        #assigns students to classrooms
        c = 0 #index for where in list room starts
        d = 0 #index for which room
        for size in room_size:
            room_list.append(np.reshape((student_numbers[c:c + size]), room_dim[d]))
            c += size #adds previous room size to room start index
            d += 1 #moves to next room
        
        # room_list is a list(ndarray): room_list[which_room][row][col] = which_student_number
        # compute neighbors, store in corresponding Student objects
        for i in range(len(room_list)):
            row_dim = room_list[i].shape[0]
            col_dim = room_list[i].shape[1]
            for row in range(row_dim):
                for col in range(col_dim):
                    student_num = room_list[i][row][col]
                    student = student_list[student_num]
                    
                    # student keeps track of their seat
                    student.add_seat((i, row, col))
        
                    # all possible combos of row +-1 and col +- 1 (with edge cases)
                    neigh_rows = list(range(max(0, row - 1), min(row + 1, row_dim - 1) + 1))
                    neigh_cols = list(range(max(0, col - 1), min(col + 1, col_dim - 1) + 1))
                    positions = set(itr.product(neigh_rows, neigh_cols))
                    positions.remove((row, col))   # remove student (student isn't neighbor of themselves)
        
                    # adding all neighbors to given student
                    for pos in positions:
                        student.add_neighbor(student_list[room_list[i][pos[0]][pos[1]]])
                        
#        # testing
#        print(room_list)
#        for i in range(len(room_list)):
#            row_dim = room_list[i].shape[0]
#            col_dim = room_list[i].shape[1]
#            for row in range(row_dim):
#                print([student_list[s].seats for s in room_list[i][row]])
#        print("\nstudent_list seats:")
#        print([s.seats for s in student_list])
#        print()

#room_assign([[4,2],[2,3],[5,4]], [Student() for i in range(34)], 2) #example

