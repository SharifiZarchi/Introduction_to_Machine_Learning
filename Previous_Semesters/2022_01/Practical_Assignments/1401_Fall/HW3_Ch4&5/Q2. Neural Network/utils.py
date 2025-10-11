import copy
import pickle
import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    def __call__(self, *args):
        return self._forward(*args)

    @abstractmethod
    def _forward(self, *args):
        pass


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def check_gradient_ce(ce, h=0.0001):
    np.random.seed(123123)
    target = np.random.randint(5, size=10)
    probs = np.random.random(size=(10, 5))
    upstream = 1

    new_probs = probs + h
    new_val = ce(new_probs, target)
    old_val = ce(probs, target)
    delta_output = new_val - old_val
    delta_loss_indirect = np.sum(delta_output * upstream)

    ce.backward(upstream)
    delta_loss_direct = np.sum(h * ce.grads['x'])

    print(f'Gradient of loss w.r.t output:\n{upstream}')
    print(f"Gradient of loss w.r.t input:\n{ce.grads['x']}")
    print(f'Relative error of delta-loss:\n{rel_error(delta_loss_indirect, delta_loss_direct)}')


def check_gradient_softmax(sm, h=0.0001):
    np.random.seed(321321)
    x = np.random.random(size=(10, 5))
    upstream = np.random.random(size=(10, 5))

    new_x = x + h
    new_val = sm(new_x)
    old_val = sm(x)
    delta_output = new_val - old_val
    delta_loss_indirect = np.sum(delta_output * upstream)

    sm.backward(upstream)
    delta_loss_direct = np.sum(h * sm.grads['x'])

    print(f'Gradient of loss w.r.t output:\n{upstream}')
    print(f"Gradient of loss w.r.t input:\n{sm.grads['x']}")
    print(f'Relative error of delta-loss:\n{rel_error(delta_loss_indirect, delta_loss_direct)}')


def check_gradient_relu(rl, h=0.0001):
    np.random.seed(11111)
    x = np.random.normal(size=(10, 5))
    upstream = np.random.random(size=(10, 5))

    new_x = x + h
    new_val = rl(new_x)
    old_val = rl(x)
    delta_output = new_val - old_val
    delta_loss_indirect = np.sum(delta_output * upstream)

    rl.backward(upstream)
    delta_loss_direct = np.sum(h * rl.grads['x'])

    print(f'Gradient of loss w.r.t output:\n{upstream}')
    print(f"Gradient of loss w.r.t input:\n{rl.grads['x']}")
    print(f'Relative error of delta-loss:\n{rel_error(delta_loss_indirect, delta_loss_direct)}')


def check_gradient_linear(linear, h=0.00001):
    np.random.seed(121212)
    x = np.random.normal(size=(10, linear.dim_in))
    upstream = np.random.random(size=(10, linear.dim_out))

    new_x = x + h
    new_w = linear.params['W'] + h
    new_b = linear.params['b'] + h
    new_linear = copy.deepcopy(linear)
    new_linear.params['W'] = new_w
    new_linear.params['b'] = new_b
    new_val = new_linear(new_x)
    old_val = linear(x)
    delta_output = new_val - old_val
    delta_loss_indirect = np.sum(delta_output * upstream)

    linear.backward(upstream)
    delta_loss_direct = np.sum(h * linear.grads['x'])
    delta_loss_direct += np.sum(h * linear.grads['W'])
    delta_loss_direct += np.sum(h * linear.grads['b'])

    print(f'Gradient of loss w.r.t output:\n{upstream}')
    print(f"Gradient of loss w.r.t input:\n{linear.grads['x']}")
    print(f"Gradient of loss w.r.t W:\n{linear.grads['W']}")
    print(f"Gradient of loss w.r.t b:\n{linear.grads['b']}")
    print(f'Relative error of delta-loss (for linear unit):\n{rel_error(delta_loss_indirect, delta_loss_direct)}')


    # check regularization gradient
    old_val = np.sum(np.power(linear.params['W'], 2))
    new_val = np.sum(np.power(new_w, 2))
    indirect_loss = new_val - old_val
    direct_loss = np.sum(h * linear.grads['reg'])
    print(f'Relative error of delta-loss (for regularization):\n{rel_error(indirect_loss, direct_loss)}')


def check_gradient_reg(linear, h=0.0001):
    np.random.seed(121212)
    upstream = np.random.random(size=(10, linear.dim_out))
    
    new_w = linear.params['W'] + h
    old_val = np.sum(np.power(linear.params['W'], 2))
    new_val = np.sum(np.power(new_w, 2))
    indirect_loss = new_val - old_val

    linear.backward(upstream)
    direct_loss = np.sum(linear.grads['reg'])

    print(f'Relative error of delta-loss:\n{rel_error(indirect_loss, direct_loss)}')


def load_dataset(train_num, test_num):
    np.random.seed(140109)
    path = './cifar-10-batches-py'
    X = list()
    Y = list()
    for i in range(5):
        with open(f'{path}/data_batch_{i+1}', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            x = data['data']
            y = data['labels']

            x = x.reshape(10000, 3, 32, 32).transpose(
                0, 2, 3, 1).astype('float')/255
            y = np.array(y)

            X.append(x)
            Y.append(y)

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    mask = np.random.choice(range(X.shape[0]), size=train_num, replace=False)
    X_train = X[mask]
    Y_train = Y[mask]

    with open(f'{path}/test_batch', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        X = data['data']
        Y = data['labels']

        X = X.reshape(10000, 3, 32, 32).transpose(
            0, 2, 3, 1).astype('float')/255
        Y = np.array(Y)

    mask = np.random.choice(range(X.shape[0]), size=test_num, replace=False)
    X_test = X[mask]
    Y_test = Y[mask]

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test
    }
