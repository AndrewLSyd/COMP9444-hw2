#!bin/bash
# submit model
scp student.py savedModel.pth z3330164@login.cse.unsw.edu.au:/import/ravel/2/z3330164/COMP9444/hw2
ssh z3330164@login.cse.unsw.edu.au
cd COMP9444/hw2
give cs9444 hw2 student.py savedModel.pth