

import random

def classes(room_dim, periods):
#room_dim = list of room dimensions for each room (e.g. seat length x seat width)
#periods = number of time periods (student has n periods (classes) a day)

    #determine number of students (seats) in each room
    room_size = []
    for n in room_dim: #for each set of dim in room_dim
        room_size.append(n[0] * n[1]) #size = r * c

    #determine total number of students, generate list of students
    num_students = sum(room_size) #total number of students
    students = range(num_students) #list of student numbers

    #assign students to classes
    class_list = []
    for i in range(periods): #repeats for each period
        random.shuffle(students) #shuffle list of students randomly

        d = 0
        for n in room_size:
            class_list.append(students[d:d + n]) #increments of students depending on room size
            d += n #keeps track of start of block, depending on size of previous room

    return(class_list) #class list includes lists of each class, repeated for different periods

print(classes([[1,2],[3,4]], 2)) #example

###Notes:
    #Need to figure out how to implement student object with determining neighbors
    #Class list is a list, need to change back to array based on room dim to figure out neighbors

