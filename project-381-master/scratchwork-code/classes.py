

import random

def classes(num_classes, students_perclass, num_period):

    num_students = students_perclass * num_classes
    students = list(range(1, num_students + 1))

    class_list = {}

    for i in range(1, num_period + 1):
        random.shuffle(students)

        for j in range(1, num_students, students_perclass):
            n = (j / students_perclass) + 1
            class_list["class{0}".format(n) + "period{0}".format(i)] = students[j:j + students_perclass]

    return(class_list)

print(classes(3, 10, 2))






