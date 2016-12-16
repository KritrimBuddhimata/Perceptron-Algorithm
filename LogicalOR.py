'''PERCEPTRON ALGORITHM ON SPECIFIED POINTS
THE POINTS ARE DEFINED AND THE EXPECTED OUTPUT IS CLASSIFICATION IN 2 CLASSES
THE POINTS ARE AS A 2 BIT DIGITAL INPUT AND THE OUTPUT IS "LOGICAL OR" OPRATION DONE ON THEM
''' 

from random import choice 
from numpy import array, dot, random 

'''Unit step function that gives 1 if parameter is non negative'''

unit_step = lambda x: 0 if x < 0 else 1 

'''
The first two entries of the NumPy array in each tuple are the two input values. 
The second element of the tuple is the expected result. 
And the third entry of the array is a "dummy" input (also called the bias) 
which is needed to move the threshold (also known as the decision boundary) up or down as needed by the step function. 
Its value is always 1, so that its influence on the result can be controlled by its weight.
'''

training_data = [ 
	(array([0,0,1]), 0), 
	(array([0,1,1]), 1), 
	(array([1,0,1]), 1), 
	(array([1,1,1]), 1), 
] 

w = random.rand(3) #Three random numbers in [0,1] as a starting point

errors = [] 

#eta is the learning rate of perceptron

eta = 0.2          												 
n = 100 
''' 
error is the list containing errors at every iteration
result is the result of classification after every iteration
'''

for i in xrange(n): 
	x, expected = choice(training_data) 
	result = dot(w, x) 
	error = expected - unit_step(result) 
	errors.append(error) 
	w += eta * error * x 

'''
formatted output and classification of the input
'''

for x, _ in training_data: 
	result = dot(x, w) 
	print("{}: {} -> {}".format(x[:2], result, unit_step(result)))