# breakout
BUILD AN AI TO BEAT THE GAME BREAKOUT
ABSTRACT:
The main Aim of our project is to improve the time complexity of the already excisting technology with the help of latest more powerful algorithms. So we are introduced A3C Asynchronous Advantage Actor-Critic algorithm which is a most powerful reinforcement learning algorithm along with LSTM layer (Long Short Term Memory). After applying these two algorithm the time taken to train the model was rapidly reduced, even there wont be any need of GPU (graphics processing units) at all just CPU is enough. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems.

INTRODUCTION:
In this paper we will mainly use A3C Asynchronous Advantage Actor-Critic We will use it to solve a simple challenge in a Breakout environment. With the holidays right around the corner, this will be my final post for the year, and I hope it will serve as a culmination of all the previous topics in the series. If you haven't yet or are new to Deep Learning and Reinforcement Learning, I have learned so much about RL and happy to share this paper with you all.
So what is A3C? The A3C algorithm was introduced by Google Deepmind group and become the most powerful reinforcement learning algorithm. It was faster, simpler, more robust, and able to achieve much better scores on the standard battery of Deep RL tasks. Given this, it has become the go-to Deep RL algorithm for new challenging problems with complex state and action spaces. OpenAI released a updated version of Asynchronous Advantage Actor-Critic.



THREE STEP PROCESS OF THE ALGORITHM:

Asynchronous: Unlike Deep Neural Network, where a single agent represented by a single neural network interacts with a single environment, A3C utilizes multiple incarnations of the above to learn more efficiently. In A3c the all different agents have their own set of network parameters Each of these agents interacts with its copy of the environment at the same time as the other agents are interacting with their environments.. In this way, the overall experience available for training becomes more diverse.
Actor-Critic: So far this series has focused on value-iteration methods such as Q-learning, or policy-iteration methods such as Policy Gradient. Actor-Critic combines the benefits of both approaches. In the case of A3C, our network will estimate both a value function V(s) (how good a certain state is to be in) and a policy π(s) (a set of action probability outputs). These will each be separate fully-connected layers sitting at the top of the network. Critically, the agent uses the value estimate (the critic) to update the policy (the actor) more intelligently than traditional policy gradient methods.
Advantage: If we think back to our  implementation of policy gradient the update rule used the discounted returns from a set of experiences in order to tell the agent which of its actions were “good” and which were “bad.” The network was then updated in order to encourage and discourage actions appropriately.







METHODOLOGY:
PROPOSED SYSTEM AND ARCHITECTURE:
The proposed model is divided into four stages they are License Extracting feature from the input, given to the Neural Network( A3C algorithm), Long Short Term Memory , optimizer.. Multiple separate environments are run in parallel, each of which contains an agent. The agents however share one neural network. Samples produced by agents are gathered in a queue, from where they are asynchronously used by a separate optimizer thread to improve the policy.

 
                                                                      Basic Implementation

The first main part is we need input in the form of PIXELS to train the model, we took a game called BREAKOUT which is a purely Pixel-based game.
                                                        
						Video input

So now we need to extract the pixels feature with the help of a deep convolutional neural network with the process of the Convolutional layer->pooling layer->flattering layer.
Here comes our implementation of the algorithm which is Asynchronous Actor-Critic Agents (A3C) developed by Google deep mind, this algorithm will use a minimum of 3 Actor(Agent), so that the three agents will start their point from different manners and according to their rewards they'll move on but the main of the 3 actors is they have a common SHARED MODEL through which they choose the optimal part of training the model little bit faster, but this fast is not enough, so we are introducing LSTM layer(Long Short Term Memory layer).

                                 
Making each move after training


The framework of A3C:
                     
                                                              The 3 As of A3C
This LSTM layer which will make a predicted move with the previous output, this part is attached to the hidden layers itself so the time that the model takes to make the next move will be very fast because the best optimal decision is taken in the hidden layer itself instead of going to the output section and then applying softmax.                            

Implementing the Algorithm Process:
A3C is currently one of the most powerful algorithms in DRL. It is basically like the smarter, cooler, hotter sibling of Deep Q networks. I was fascinated by the capabilities of this algorithm so I implemented it to learn and play a game called Breakout. My aim is to make it play same like a human

 




A3C Brain:
 This is the most important of the project, this contains all A3C algorithms and the LSTM layer implemented. 
 

And finally, the trained model can be seen through each episode length. I have totally used 10000 episode length 
The video will be our output which is automatically trained with the concept of Reinforcement Learning through the A3C algorithm,  in this, there will be seven videos which will be generated at the end of 10000 episode  






DATASET :
To train and test the model we used a Breakout Environment video input, the input image is given below.
The input can be extracted by video image PIXELS. Breakout environment outputs 210*160 RCB arrays (210*160*3).

 
the convolutional neural network we have to implement the A3C algoritm with 3 Agent ,we can have more than three Actor as of now we are just creating the 3 Agent to working simultaneously by using the SHARED MODEL

 
                                                    Shared Critic model
so this is how how the model is training faster than the other model but when we include this algorithm alone the result wasn’t that great to say because it took a longer time to train this is not the optimal part of it so when we came to see the backend of how it works in order to solve this problem .
we came to know that this feature extraction works on the principle of taking pixels from each movement for your better understanding i have enclosed the image below,
                                                     
fig 1
as we see in the (fig 1) we can see that the next move will be left downwards but the AI dont know what to do, it will only come to know that it is moving left downwards after many softamx functions.



EXPERIMENTAL SETUP :
The experiment was carried out in spyder(Anaconda) with a new Conda environment and Pytorch, gym, OpenCV installed with it. The laptop configuration is intel i7, 8GB RAM without any usage of GPU. This video took ~9hrs to train the Game.


EXPERIMENTAL ANALYSIS :
Several experiments were carried out to get a better model that would be more generalized in all cases. But no one has thought of improving the time complexity of the training the model, We came up with the idea of using the google deepmind algorithm to improvise it.

PERFORMANCE :
	The Breakout we trained with the same parameters presented below
 

It usually takes about ~40hrs on Tesla K80 GPU or ~90h on 2.9 GHz Intel i7 Quad-Core CPU. But for our code took only ~9hrs with 8GB RAM i7 configuration 
The model training visualization with a max episode length of 10000:

 
	       Breakout A3C (LSTM Layer) training
Advantages:
•	This algorithm is faster and more robust than the standard Reinforcement Learning Algorithms.
•	It performs better than the other Reinforcement learning techniques because of the diversification of knowledge as explained above.
•	It can be used on discrete as well as continuous action spaces.


Here is the final performance of our trained model which plays the game perfectly.
 
Final Result

CONCLUSION AND FUTURE SCOPE
The main part of our project compared with others where the time is taken to train the model was 70% reduced! ( from 40hrs to ~9hrs ). This algorithm can be used in all computer vision or video-based real-time projects, mainly on Self-driving cars because they need to make a positive decision in a short period. 

REFERENCES:
	https://medium.com/analytics-vidhya/how-i-built-an-algorithm-to-takedown-atari-games-a13d3b3def69
https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f
Barto, Andrew G, Sutton, Richard S, and Anderson, Charles W. Neuronlike adaptive elements that can solve difficult learning control problems. Systems, Man and Cybernetics, IEEE Transactions on, (5):834–846, 1983.

Baxter, Jonathan, and Bartlett, Peter L. Reinforcement learning in POMDPs via direct gradient ascent. In ICML, pp. 41–48.
 Bertsekas, Dimitri P. Dynamic programming and optimal control, volume 2. Athena Scientific.
 Bhatnagar, Shalabh, Precup, Doina, Silver, David, Sutton, Richard S, Marie, Hamid R, and Szepesvari, Csaba. ´ Convergent temporal-difference learning with arbitrary smooth function approximation. In Advances in Neural Information Processing Systems, pp. 1204–1212. 
 Greensmith, Evan, Bartlett, Peter L, and Baxter, Jonathan. Variance reduction techniques for gradient estimates in reinforcement learning. The Journal of Machine Learning Research, 5:1471–1530.
 Hafner, Roland, and Riedmiller, Martin. Reinforcement learning in feedback control. Machine learning, 84 (1-2):137–169. 
 He, Nicolas, Wayne, Greg, Silver, David, Lillicrap, Timothy, Tassa, Yuval, and Erez, Tom. Learning continuous control policies by stochastic value gradients. arXiv preprint arXiv:1510.09142. 
 Hull, Clark. Principles of behavior.
Nicolas, Erez, Tom, Tassa, Yuval, Silver, David, and Wierstra, Daan. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.


