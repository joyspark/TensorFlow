import tensorflow as tf
import numpy as np

tf.set_random_seed(777)# for reproducibility

# Parameters
learning_rate = 0.3
training_epochs = 500

# real data
data = np.loadtxt('./data/train.csv', delimiter=',', dtype=np.int64).astype(np.float32)
input = data[:,:-1]
output = data[:,-1].reshape([len(data),1])
features = input.shape[1]

test_data = np.loadtxt('./data/test.csv', delimiter=',', dtype=np.int64).astype(np.float32)
test_data = test_data.reshape([-1, features]) # make uniform shape

print (input.shape, output.shape)

# placeholders for train
X = tf.placeholder(tf.float32, shape=[None, features], name='x')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

# Set model weights
W = tf.Variable(tf.random_normal([features, 1],dtype='float'), name='weight')
b = tf.Variable(tf.random_normal([1],dtype='float'), name='bias')

# Construct a linear model
hypothesis = tf.matmul(X, W, name='h') + b
h = tf.identity(hypothesis, name='h')

# Mean squared error
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# accuracy
acc = tf.equal(tf.round(hypothesis), Y)
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(training_epochs):
	sess.run(train, feed_dict={X: input, Y: output})
	if(step%100==0):
		cost_val, hy_val = sess.run([cost, hypothesis], feed_dict={X: input, Y: output})
		if step==0:
			print("[",step+1,"] Cost:", cost_val)
		else:
			print("[",step,"] Cost:", cost_val)

print("=============TRAIN END==============")
for i in range(len(data)):
	hy_val = sess.run(hypothesis, feed_dict={X: input[i,:].reshape([1,features])})
	print(" Answer:", output[i], " Prediction:", round(hy_val[0,0]))

cost_val, acc_val = sess.run([cost,acc], feed_dict={X: input, Y: output})
print(" Cost:", cost)
print(" Acc:", acc_val)

print("=============PREDICT==============")
for i in range(test_data.shape[0]):
	pre = sess.run(hypothesis, feed_dict={X: test_data[i].reshape(1,features)})
	print(test_data[i],"==> ", pre[0])

builder = tf.saved_model.builder.SavedModelBuilder("/tmp/fromPython")
builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
builder.save()
