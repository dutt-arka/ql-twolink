import matplotlib.pyplot as plt
with open ('data.txt','r') as f:
    lines=f.readlines()
    x= [float(line.split()[0]) for line in lines]
    y= [float(line.split()[1]) for line in lines]

plt.title('rewards over episodes')    
plt.xlabel('episodes')
plt.ylabel('reward')

plt.plot(x ,y)
plt.savefig('foo.png')
plt.show()

