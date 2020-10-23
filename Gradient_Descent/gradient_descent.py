import numpy as np

# we have correct values of x and y and we want to come up with correct
# values of m and b
# expected value of m = 2 and b = 3
def gradient_descent(x,y):
    m_curr = b_curr = 0
    # we start with some values and take baby steps to reach global minimum
    iterations = 10000
    # length of datapoints
    n = len(x)
    # start with some learning rate  (trial and error to see how your algo behaves)
    learning_rate = 0.08
    # it is like a trial and error process
    # this is the process of moving towards the best fit line
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        # at each step we should be reducing the cost function
        cost = (1 / n) * sum([val**2 for val in (y-y_predicted)])
        # m partial derivative
        md = -(2 / n) * sum(x * (y-y_predicted))
        # b partial derivative
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {},b {}, iteration {}, cost {}".format(m_curr,b_curr,i,cost))


# for faster matrix multiplication
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)