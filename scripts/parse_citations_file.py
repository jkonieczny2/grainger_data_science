#!/usr/bin/env python
# coding: utf-8

# Open files to write corrupted and uncorrupted data
corrupted_filename = '../data/parking_citations_corrupted.csv'
uncorrupted_filename = '../data/parking_citations_uncorrupted.csv'

corrupt = open(corrupted_filename , 'w')
uncorrupt = open(uncorrupted_filename , 'w')

# Loop full CSV, routing rows to corrupted and uncorrupted files
with open('../data/parking_citations.corrupted.csv', 'r') as f:
    reader = f.readlines()
    header = reader[0]
    corrupt.write(header)
    uncorrupt.write(header)
    
    for line in reader[1:]:
        test = line.split(',')[8]
        
        if test == "" :
            corrupt.write(line)
        else:
            uncorrupt.write(line)

            
corrupt.close()
uncorrupt.close()

