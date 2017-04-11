import numpy as np

# assume gradients are normally distributed


epochs = 1000
grads = np.random.normal(size = 10000)
gradients, approx = [], []
for i in range(epochs):
	grads = np.random.normal(size = 10000)
	expected_gradient = np.mean(grads)
	gradients.append(expected_gradient)
	stochastic_sample = np.random.choice(grads, 10)
	sgd = np.mean(stochastic_sample)
	approx.append(sgd)

gradients, approx = np.array(gradients), np.array(approx)
print("vanilla gd mean: {} and var: {}".format(np.mean(gradients), np.var(gradients)))
print("sgd mean: {} and var: {}".format(np.mean(approx), np.var(approx)))

# print(stochastic_sample)