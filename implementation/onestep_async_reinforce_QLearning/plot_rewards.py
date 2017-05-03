import pickle
import matplotlib.pyplot as plt
global_rewards = pickle.load(open("Rewards_onestep_SARSA_Binary.txt", 'rb'))

x = []
y = []

print (global_rewards)
'''
keylist = global_rewards.keys()
keylist = sorted(keylist)
for key in keylist:
	reward = global_rewards[key]
	if reward != 0:
		x.append(key)
		y.append(reward)



plt.plot(y, x, marker='.')
plt.show()

#plt.plot(x, y, 'ro')
#plt.show()
'''
