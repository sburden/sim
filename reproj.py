
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2013 

import os
import sys
import time

import numpy as np
import pylab as plt
import matplotlib as mpl
import scipy as sp
import scipy.optimize as op

np.set_printoptions(precision=6)
mpl.rc('legend',fontsize=14)

import relax as rx

def tgt(f,x,d=1e-6,Map=map):
  """
  U = tgt  Approximate tangent space

  Inputs
    f : R^n --> R^m - submanifold coordinates
    x \in R^n
  (optional)
    d - scalar - diameter of `tangent' neighborhood

  Outputs
    U - m x n - columns span T_f(x) f(R^n)
  """
  x = np.array(x).flatten(); n = x.size;
  # apply f to points near x
  #X = x + d*(np.random.rand(2*n,n)-0.5)
  fx = f(x); m = fx.size
  X = x + np.vstack((d*np.identity(n),
                     -d*np.identity(n),
                     d*(np.random.rand(2*n,n)-0.5)))
  Z = np.array(Map(f,X))
  # find orthonormal basis for tangent space T_f(x) f(R^m)
  U,_,_ = np.linalg.svd(np.cov(Z.T))
  U = U[:,:n]

  return U

def jac(f,x,d=1e-6,Map=map):
  """
  Df = jac  Approximate Jacobian of map

  Inputs
    f : R^n --> R^m - submanifold coordinates
    x \in R^n
  (optional)
    d - scalar - diameter of `tangent' neighborhood

  Outputs
    Df - m x n - Jacobian; columns span T_f(x) f(R^n)
  """
  x = np.array(x).flatten(); n = x.size;
  # apply f to points near x
  #X = x + d*(np.random.rand(2*n,n)-0.5)
  fx = f(x); m = fx.size
  X = x + np.vstack((d*np.identity(n),
                     -d*np.identity(n)))
  fX = np.array(Map(f,X))
  Df = (fX[:n,:] - fX[n:,:]).T / (2*d)

  return Df

def proj(v,U):
  """
  u = proj  Project vector orthogonally onto column span

  Inputs
    v - n x 1 - vector to project
    U - n x m - columns span subspace 

  Outputs
    u := U (U^T U)^{-1} U^T v
  """
  U = np.matrix(U)
  n,m = U.shape
  v = np.matrix(v).reshape(n,1)
  u = np.array((U*((U.T*U).I)*U.T)*v)

  return u

def reproj(f,z,dz,x0,Df=None,d=1e-6,Map=map):
  """
  x,dx,Df = proj  Project state, tangent vector onto submanifold

  Inputs
    f : R^n --> R^m - submanifold coordinates
    z \in R^m, dz \in T_z R^m
    x0 \in R^n - initial guess
  (optional)
    Df : TR^n --> TR^m - Jacobian of f
    d - scalar - diameter of `tangent' neighborhood
    s - int - number of samples for approximating tangent space

  Outputs
    x  := arg min { |f(x) - z|^2 : x \in R^n}
    dx := U (U^T U)^{-1} U^T dz; cols of U span T_f(x) f(R^n) 
    x \in R^n, dx \in T_f(x) f(R^n)
  """
  x0 = np.array(x0).flatten(); n = x0.size
  z  = np.array(z).flatten();  m = z.size
  # project z into f(R^m)
  x,_ = op.leastsq(lambda y : f(y).flatten() - z,x0,Dfun=Df)
  x = np.array(x).reshape(n)
  # project dz into T_x R^m
  # dx = (U.T U)^{-1} U^T dz
  dz = np.matrix(dz).reshape(m,1)
  if Df is None:
    Df = np.matrix(jac(f,x,Map=Map))
  dx = np.linalg.inv(Df.T * Df) * Df.T * dz
  dx = dx / np.linalg.norm(Df * dx)
  dx = np.array(dx).flatten()

  return x,dx,Df


if __name__ == "__main__":

  args = sys.argv

  if '-p' in args:
    print 'Setting up parallel computations:',
    sys.stdout.flush(); st = time.time()
    from IPython.kernel import client
    mec = client.MultiEngineClient()
    mec.activate()
    mec.run(args[0])
    Map = mec.map
    print '%0.2f sec' % (time.time()-st)

  else:
    Map = map

  f = lambda x : np.array([x[0],x[0]**2])
  f = lambda x : np.array([np.cos(x),np.sin(x)]).flatten()
  #def f(x):
  #  if np.sin(x) >= 0.:
  #    return np.array([np.cos(x),np.sin(x)]).flatten()
  #  else:
  #    return np.array([np.cos(x),np.sin(x),0.*x]).flatten()
  X0 = np.linspace(0.,2*np.pi)
  fX0 = np.array(map(f,X0))

  x0 = [0.]#np.random.randn(1)
  z  = f(x0)# + np.hstack([np.random.randn(1)*0.1,0.])
  dz = jac(f,x0).flatten()
  #if dz[1] < 0:
  #  dz = -1.*dz

  x,dx,Df = reproj(f,z,dz,x0,Map=map)

  t0 = 0.; tf = 2*np.pi
  n = 10
  T  = np.linspace(t0,tf,n)
  X  = np.ones((T.size,1))*x
  dX = np.ones((T.size,1))*dx
  Z  = np.ones((T.size,1))*z
  dZ = np.ones((T.size,1))*dz
  Df = np.ones((T.size,len(z),len(x)))

  for k in range(1,len(T)):
    Z[k]  = f(X[k-1]) + dZ[k-1]*(T[k]-T[k-1])
    X[k],dX[k],Df[k] = reproj(f,Z[k],dZ[k-1],X[k-1],Map=Map)
    dZ[k] = np.dot(Df[k],dX[k][:,np.newaxis]).flatten()

  fX = np.array(map(f,X))

  plt.ion()
  plt.figure(1); plt.clf()
  ms = 10.; dms = ms/2.
  lw = 2.

  ax = plt.subplot(2,1,1); ax.grid('on'); ax.axis('equal')
  ax.plot(fX0[:,0],fX0[:,1],'k')
  ax.plot(fX[:,0],fX[:,1],'k.',ms=ms,label='$x$')
  ax.plot(Z[:,0],Z[:,1],'b',ms=ms,label='$z$')
  ax.plot(np.vstack((Z[:,0],Z[:,0]+dZ[:,0])),
          np.vstack((Z[:,1],Z[:,1]+dZ[:,1])),'b.-',
          ms=ms)
  ax.legend()

  ax = plt.subplot(2,1,2); ax.grid('on');
  dZn = np.array(map(np.linalg.norm,dZ))
  ax.plot(dZn,'.-',lw=lw,ms=ms,label='$\\left|\dot{z}\\right|$')
  ax.set_ylim(0,2.)
  ax.legend()

  plt.draw()

