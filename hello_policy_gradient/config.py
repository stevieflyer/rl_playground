"""
The configuration file for the policy gradient algorithm
"""

H = 200
"""number of hidden layer neurons"""

D = 80 * 80
"""input dimensionality: 80x80 grid"""

batch_size = 10
"""every how many episodes to do a param update?"""

learning_rate = 1e-4
"""learning rate"""

gamma = 0.99
"""discount factor for reward"""

decay_rate = 0.99
"""decay factor for RMSProp leaky sum of $grad^2$. RMSProp is a kind of optimization 
algorithm. Generally speaking, the larger decay rate, the less important the update history 
is. For more math details, you can refer to [this blog](https://zhuanlan.zhihu.com/p/34230849)."""

resume = False
"""resume from previous checkpoint?"""

render = False
"""whether to render the environment"""
