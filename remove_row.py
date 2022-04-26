import tensorflow as tf

sess = tf.InteractiveSession()

matrix1 = tf.constant([[1., 2., 3., 4., 5., 6.], [3., 1., 2., 2., 0., 1.]])
matrix2 = tf.constant([[2., 0., 1., 9., 3., 2.5], [1.8, 2.5, 9.4, 1., 0.3, 3.1]])

#lets make a 6x6 matrix by append
A = tf.concat([matrix2], 0) #Now A is a 2 x 6 tensor
A = tf.concat([A, matrix1], 0) #Now A is a 4 x 6 tensor
A = tf.concat([A, matrix1], 0) #Now A is a 6 x 6 tensor
print(A.eval()) #Lets print it

num_row_A = A.get_shape().as_list()[0] #we need these to remove (slice)
num_col_A = A.get_shape().as_list()[1]

B = tf.slice(A, [1,0], [num_row_A-1, num_col_A]) #the second argument of the function is offset for beginning the slice, the third argument is the shape of the resulting matrix

print(B.eval()) #B will be A with first row removed

#close the session to release resources
sess.close()
