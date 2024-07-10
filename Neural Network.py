#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:33:32 2023

@author: maharsh
"""
import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt


print("------------------------EX1-----------------------------------------")


np.random.seed(1)
input_maharsh=np.random.uniform(-0.6,0.6,(10,2))
print(input_maharsh)

output_maharsh = np.sum(input_maharsh, axis=1).reshape((-1, 1))
#print(output_maharsh)
print(np.min(input_maharsh[:,0]))
print(np.max(input_maharsh[:,1]))

print([np.min(input_maharsh), np.max(input_maharsh)], [np.min(input_maharsh), np.max(input_maharsh)])
print([[np.min(input_maharsh), np.max(input_maharsh)], [np.min(input_maharsh), np.max(input_maharsh)]])
network = nl.net.newff([[np.min(input_maharsh[:,0]), np.max(input_maharsh[:,0])],[np.min(input_maharsh[:,1]), np.max(input_maharsh[:,1])]], [6, 1])


error = network.train(input_maharsh, output_maharsh, show=15, goal=0.00001)


result1 = network.sim([[0.1, 0.2]])

print(f"result 1: {result1}")


print("------------------------EX2-----------------------------------------")



network2 = nl.net.newff([[np.min(input_maharsh[:,0]), np.max(input_maharsh[:,0])],[np.min(input_maharsh[:,1]), np.max(input_maharsh[:,1])]], [5,3,1])

network2.trainf = nl.train.train_gd

error2 = network2.train(input_maharsh, output_maharsh,epochs=1000, show=100, goal=0.00001)


result2 = network2.sim([[0.1, 0.2]])

print(f"result 2: {result2}")


print("------------------------EX3-----------------------------------------")


np.random.seed(1)
input_maharsh=np.random.uniform(-0.6,0.6,(100,2))
print(input_maharsh)


output_maharsh = np.sum(input_maharsh, axis=1).reshape((-1, 1))
print(output_maharsh)


network3 = nl.net.newff([[np.min(input_maharsh[:,0]), np.max(input_maharsh[:,0])],[np.min(input_maharsh[:,1]), np.max(input_maharsh[:,1])]], [6, 1])


error = network3.train(input_maharsh, output_maharsh, show=15, goal=0.00001)


result3 = network3.sim([[0.1, 0.2]])

print(f"result 3: {result3}")



print("------------------------EX4-----------------------------------------")



network4 = nl.net.newff([[np.min(input_maharsh[:,0]), np.max(input_maharsh[:,0])],[np.min(input_maharsh[:,1]), np.max(input_maharsh[:,1])]], [5,3,1])

network4.trainf = nl.train.train_gd

error4 = network4.train(input_maharsh, output_maharsh,epochs=1000, show=100, goal=0.00001)

plt.plot(error4)
plt.xlabel("Number of epochs")
plt.ylabel("Training error")
plt.grid()
plt.show()
result4 = network4.sim([[0.1, 0.2]])

print(f"result 4: {result4}")


print("------------------------EX5-----------------------------------------")


np.random.seed(1)
input_maharsh=np.random.uniform(-0.6,0.6,(10,3))
print(input_maharsh)

output_maharsh = np.sum(input_maharsh, axis=1).reshape((-1, 1))
print(output_maharsh)

network5 = nl.net.newff([[np.min(input_maharsh[:,0]), np.max(input_maharsh[:,0])],[np.min(input_maharsh[:,1]), np.max(input_maharsh[:,1])],[np.min(input_maharsh[:,2]), np.max(input_maharsh[:,2])]], [6, 1])


error5 = network5.train(input_maharsh, output_maharsh, show=15, goal=0.00001)


result5 = network5.sim([[0.2,0.1, 0.2]])

print(f"result 5: {result5}")

network6 = nl.net.newff([[np.min(input_maharsh[:,0]), np.max(input_maharsh[:,0])],[np.min(input_maharsh[:,1]), np.max(input_maharsh[:,1])],[np.min(input_maharsh[:,2]), np.max(input_maharsh[:,2])]], [5,3,1])

network6.trainf = nl.train.train_gd

error6 = network6.train(input_maharsh, output_maharsh,epochs=1000, show=100, goal=0.00001)

result6 = network6.sim([[0.2,0.1, 0.2]])

print(f"result 6: {result6}")

print("------------------------All results-----------------------------------------")


print(f"result 1: {result1}   10 data points and structure(6,1)")
print(f"result 2: {result2}   10 data points and structure(5,3,1)")
print(f"result 3: {result3}   100 data points and structure(6,1)")
print(f"result 4: {result4}   100 data points and structure(5,3,1)")
print(f"result 5: {result5}   10 data points, 3 inputs and structure(6,1)")
print(f"result 6: {result6}   10 data points, 3 inputs and structure(5,3,1)")








