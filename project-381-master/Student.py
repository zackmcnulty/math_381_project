#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:42:07 2018

@author: KCR
"""

class Student:
    def __init__(self):
        # all possible states (subject to change)
        # state 0 = susceptible/uninfected
        # state 2 = infected
        # state 3 = recovered (immune)
        self.state = 0
        self.days_infected = list()
        self.was_vaccinated = False
        
        # set of student's neighbors throughout the day
        self.neighbors = set()
        # set of (room, row, col) the student is in during the day
        self.seats = list()

        # determined using geometric distribution
        self.stays_sick_for = 10  # 10 is placeholder for now

    # toString method
    def __repr__(self):
        return "Student(" + str(self.state) + ")"

    def set_recovery_time(self, days_sick):
        self.stays_sick_for = days_sick

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state
        
    def set_vaccinated(self):
        '''Records that the student was vaccinated'''
        self.was_vaccinated = True

    def add_neighbor(self, other):
        self.neighbors.add(other)
        
    def add_seat(self, seat):
        '''Given seat = (room_index, row, column), adds that seat to student's
        list of seats visited throughout the day'''
        self.seats.append(seat)

    def get_neighbors(self):
        return self.neighbors

    def add_day_infected(self, day):
        self.days_infected.append(day)

    def get_days_infected(self):
        return self.days_infected
