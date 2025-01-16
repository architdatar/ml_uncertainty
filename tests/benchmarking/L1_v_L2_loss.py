"""
To get Hessian matrix of loss function, we need to take its derivative.
Somehow, Autograd does not take the derivative of the numpy L1 norm.
This is likely because it is a modulus function the derivative of which
is not defined at x=0.

So, to estimate this, we have instead used the L2 norm and its derivatives.
In this example, we see whether that assumption is justified.
"""
#%%
# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import hessian


def l1_norm(params):
    """Actual L1 norm"""

    return np.linalg.norm(params, ord=1)


def l2_norm(params):
    """L2 norm"""

    return np.linalg.norm(params, ord=2)


# x = np.arange(-100, 100)
x = np.array([-0.0001, 0.15])

# Taking the Jacobians in each case:
# For L1: jac(x) = [sign(x1) sign(x2) ... sign(xn)]

# L2: 1/L2 * [x1 x2 ... xn]

# Plot the first derivative arrays
plt.figure()
plt.plot(x, np.sign(x), color="red", label=r"$\nabla_x L1 norm$")
plt.plot(x, x / l2_norm(x), color="blue", label=r"$\nabla_x L2 norm$")
plt.legend(loc="lower right")
# Taking the Hessians w.r.t parameters in each case:
# For L1 : 0_nxn

# For L2: [... d (xi/L2) / dxi dxj ...]
hessian_mat = hessian(lambda params: np.linalg.norm(params, 2))(x)

# See how different this matrix is from 0.
print("Det: " + str(np.linalg.det(hessian_mat - np.zeros((x.shape[0], x.shape[0])))))
print("Max: " + str(np.max(hessian_mat - np.zeros((x.shape[0], x.shape[0])))))

"""
Conclusions:
1. For all arrays, the first derivative of the L1 and L2 norms
    differ significantly, but the second derivatives are very close
    to each other.
2. Thus, if the objective is to compute the Hessian matrix, the
    errors with either of these approaches are exceedingly small.
"""
# %%
