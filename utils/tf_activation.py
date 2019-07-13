""" 

Customized tensorflow activation function
Author: Wang Qinyi
Date: Sept 2018
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape

vth = 0.2
a = 2.5
tau = 0.1


#------------------- define activation fuction -----------------------------------
def spiky_2(u):
    return tf.cond(tf.less(tf.nn.relu(u), 0),tf.zeros(), lambda: 1)



def spiky(u):
    return np.where(u < vth, 0., 1.)

def gate(o):
    return np.where(o == 1., 0, tau)

#------------------------ define gradient ---------------------------------------- 
def d_spiky_rectangular(u):
    if abs(u-vth)<a/2:
        return 1/a
    else: 
        return 0.

def d_spiky_polynomial(u):
    if np.sqrt(a)/2-abs(u-vth)>0:
        return 2/np.sqrt(a)-(4*abs(u-vth))/a
    else: 
        return 0.

def d_spiky_sigmoid(u):
    e = np.exp((vth-u)/a)
    return e/np.sqrt(2*np.pi*a)

def d_spiky_guassian(u):
    e = np.exp(-(u-vth)*(u-vth)/(2*a))
    return e/np.sqrt(2*np.pi*a)
     
def d_gate(o):
    return -np.exp(-o/tau)

#----------------------- make a numpy function ------------------------------------ 
np_spiky = np.vectorize(spiky)
np_gate = np.vectorize(gate)

np_d_spiky_rectangular = np.vectorize(d_spiky_rectangular)
np_d_spiky_polynomial = np.vectorize(d_spiky_polynomial)
np_d_spiky_sigmoid = np.vectorize(d_spiky_sigmoid)
np_d_spiky_guassian = np.vectorize(d_spiky_guassian)
np_d_gate = np.vectorize(d_gate)

#------------------- make a numpy fct to a tensorflow -----------------------------
np_spiky_32 = lambda u: np_spiky(u).astype(np.float32)
np_gate_32 = lambda u : np_gate(u).astype(np.float32)

np_d_spiky_rectangular_32 = lambda u: np_d_spiky_rectangular(u).astype(np.float32)
np_d_spiky_polynomial_32 = lambda u: np_d_spiky_polynomial(u).astype(np.float32)
np_d_spiky_sigmoid_32 = lambda u: np_d_spiky_sigmoid(u).astype(np.float32)
np_d_spiky_guassian_32 = lambda u: np_d_spiky_guassian(u).astype(np.float32)
np_d_gate_32 = lambda u: np_d_gate(u).astype(np.float32)

#----------------------------------------------------------------------------------

def tf_d_spiky_rectangular(u,name=None):
    with tf.name_scope(name, "d_spiky_rectangular", [u]) as name:
        y = tf.py_func(np_d_spiky_rectangular_32,
                        [u],
                        [tf.float32],
                        name=name,
                        stateful=False)
        y[0].set_shape(u.shape)
        return y[0]

def tf_d_spiky_polynomial(u,name=None):
    with tf.name_scope(name, "d_spiky_polynomial", [u]) as name:
        y = tf.py_func(np_d_spiky_polynomial_32,
                        [u],
                        [tf.float32],
                        name=name,
                        stateful=False)
        y[0].set_shape(u.shape)
        return y[0]

def tf_d_spiky_sigmoid(u,name=None):
    with tf.name_scope(name, "d_spiky_sigmoid", [u]) as name:
        y = tf.py_func(np_d_spiky_polynomial_32,
                        [u],
                        [tf.float32],
                        name=name,
                        stateful=False)
        y[0].set_shape(u.shape)
        return y[0]

def tf_d_spiky_guassian(u,name=None):
    with tf.name_scope(name, "d_spiky_guassian", [u]) as name:
        y = tf.py_func(np_d_spiky_polynomial_32,
                        [u],
                        [tf.float32],
                        name=name,
                        stateful=False)
        y[0].set_shape(u.shape)
        return y[0]

def tf_d_gate(o, name=None):
    with tf.name_scope(name, "d_gate", [o]) as name:
        y = tf.py_func(np_d_gate_32,
                        [o],
                        [tf.float32],
                        name=name,
                        stateful=False)
        y[0].set_shape(o.shape)
        return y[0]


#----------------------------------------------------------------------------

def spikygrad_rectangular(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_spiky_rectangular(x)
    return grad * n_gr 

def spikygrad_polynomial(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_spiky_polynomial(x)
    return grad * n_gr 

def spikygrad_sigmoid(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_spiky_sigmoid(x)
    return grad * n_gr 

def spikygrad_guassian(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_spiky_guassian(x)
    return grad * n_gr 

def gategrad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_gate(x)
    return grad * n_gr 


#------------------------------------------------------------------------------
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


# Now we are almost done, the only thing is that the grad function we need to pass to the 
# above py_func function needs to take a special form. It needs to take in an operation, 
# and the previous gradients before the operation and propagate the gradients backward 
# after the operation.

def tf_spiky_rectangular(u, name=None):
    with tf.name_scope(name, "spiky_rectangular", [u]) as name:
        y = py_func(np_spiky_32,
                    [u],
                    [tf.float32],
                    name=name,
                    grad=spikygrad_polynomial)  # <-- here's the call to the gradient
        y[0].set_shape(u.shape)
        return y[0]

def tf_spiky_polynomial(u, name=None):
    with tf.name_scope(name, "spiky_polynomial", [u]) as name:
        y = py_func(np_spiky_32,
                    [u],
                    [tf.float32],
                    name=name,
                    grad=spikygrad_polynomial)  # <-- here's the call to the gradient
        y[0].set_shape(u.shape)
        return y[0]

def tf_spiky_sigmoid(u, name=None):
    with tf.name_scope(name, "spiky_sigmoid", [u]) as name:
        y = py_func(np_spiky_32,
                    [u],
                    [tf.float32],
                    name=name,
                    grad=spikygrad_polynomial)  # <-- here's the call to the gradient
        y[0].set_shape(u.shape)
        return y[0]

def tf_spiky_guassian(u, name=None):
    with tf.name_scope(name, "spiky_guassian", [u]) as name:
        y = py_func(np_spiky_32,
                    [u],
                    [tf.float32],
                    name=name,
                    grad=spikygrad_polynomial)  # <-- here's the call to the gradient
        y[0].set_shape(u.shape)
        return y[0]

def tf_gate_(o, name=None):
    with tf.name_scope(name, "gate", [o]) as name:
        y = py_func(np_gate_32,
                    [o],
                    [tf.float32],
                    name=name,
                    grad=gategrad)  # <-- here's the call to the gradient   
        y[0].set_shape(o.shape)
        return y[0]
 
#---------------------------------------------------------------------------------
def tf_spiky(u, vth, a, gradappr, name=None):
    if gradappr == 'rectangular':
        return tf_spiky_rectangular(u)
    if gradappr == 'polynomial':
        return tf_d_spiky_polynomial(u)
    if gradappr == 'sigmoid':
        return tf_spiky_sigmoid(u)
    if gradappr == 'guassian':
        return tf_spiky_guassian(u)
    
    

def tf_gate(o, tau, name=None):
    return tf_gate_(o)




if __name__  == '__main__':
    gradappr = 'sigmoid'
    with tf.Session() as sess:
        u = tf.constant([[0.2, 1.2, 1.5, 1.0],[0.8, 1.0, 1.2, 0.5]], dtype = tf.float32)
        o = tf_spiky(u, vth, a, gradappr, name=None)
        print(o.eval())
        print(tf.gradients(o, [u])[0])
        print(tf.gradients(o, [u])[0].eval())
        f = tf_gate(o, tau)
        print(f.eval())
        print(tf.gradients(f, [o])[0])
        print(tf.gradients(f, [o])[0].eval())
