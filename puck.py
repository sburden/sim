
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

import numpy as np
import pylab as plt
import matplotlib as mpl

import os
import time

import sim.hds as hds

from opt import opt

from util import Struct
from util import num

font = {'family' : 'sans-serif',
        'size'   : 18}

mpl.rc('font', **font)

np.set_printoptions(precision=2)
	
class Puck(hds.HDS):
  def __init__(self,dt=1./500):
    """
    Puck  Puck hybrid system
    """
    super(Puck, self).__init__(dt=dt)
    
    self.name = 'Puck'
    self.accel = lambda t,x,q : np.zeros((x.shape[0],3))
      
  def dyn(self, t, x, q):
    """
    .dyn  evaluates system dynamics
    """
    p = q.copy()
    # perturbation
    acc = self.accel(t,np.array([x]),q).flatten()
    # unpack state
    x,y,theta,dx,dy,dtheta = x
    # Cartesian dynamics
    dx = [dx, dy, dtheta, acc[0], acc[1], acc[2]]
    # return vector field
    return dx

  def obs(self):
    """
    .obs  observes trajectory
    """
    o = hds.Obs(t=self.t,x=[],y=[],theta=[],dx=[],dy=[],dtheta=[],
                         v=[],delta=[],q=[],acc=[])

    for t,x,q in zip(self.t,self.x,self.q):
      # perturbation
      acc = self.accel(t,x,q)
      # pre-allocate
      ones = np.ones((x.shape[0],1))
      # unpack state
      x,y,theta,dx,dy,dtheta = x.T
      # discrete mode 
      q = q*ones
      # invariant state
      v = np.sqrt(dx**2 + dy**2)
      delta = np.angle(np.exp(-1j*theta)*(dy + 1j*dx))
      # append observation data
      o.x += [np.c_[x]]
      o.y += [np.c_[y]]
      o.theta += [np.c_[theta]]
      o.dx += [np.c_[dx]]
      o.dy += [np.c_[dy]]
      o.dtheta += [np.c_[dtheta]]
      o.v += [np.c_[v]]
      o.delta += [np.c_[delta]]
      o.q += [np.c_[q]]
      o.acc += [np.c_[acc]]
    # store result
    self.o = o
    return self.o

  def evts(self, q):
    """
    .evts  returns event functions for given hybrid domain
    """
    return [lambda t,x : 1.]

  def anim(self, o=None, dt=1e-3, fign=1):
    """
    .anim  animate trajectory

    INPUTS:
      o - Obs - trajectory to animate

    OUTPUTS:
    """
    if o is None:
      o = self.obs().resample(dt)

    t = np.hstack(o.t)
    x = np.vstack(o.x)
    y = np.vstack(o.y)
    theta = np.vstack(o.theta)
    dtheta = np.vstack(o.dtheta)
    v = np.vstack(o.v)
    delta = np.vstack(o.delta)

    te = np.hstack(o.t[::2])
    xe = np.vstack(o.x[::2])
    ye = np.vstack(o.y[::2])
    thetae = np.vstack(o.theta[::2])
 
    z = np.array([v[-1],delta[-1],theta[-1],dtheta[-1]])

    def Ellipse((x,y), (rx, ry), N=20, t=0, **kwargs):
      theta = 2*np.pi/(N-1)*np.arange(N)
      xs = x + rx*np.cos(theta)*np.cos(-t) - ry*np.sin(theta)*np.sin(-t)
      ys = y + rx*np.cos(theta)*np.sin(-t) + ry*np.sin(theta)*np.cos(-t)
      return xs, ys

    r = 1.01

    mx,Mx,dx = (x.min(),x.max(),x.max()-x.min())
    my,My,dy = (y.min(),y.max(),y.max()-y.min())
    dd = 5*r

    fig = plt.figure(fign,figsize=(5*(Mx-mx+2*dd)/(My-my+2*dd),5))
    plt.clf()
    ax = fig.add_subplot(111,aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])

    Lcom, = ax.plot(x[0], y[0], 'b.', ms=10.)
    Ecom, = ax.plot(*Ellipse((x[0],y[0]), (r, 0.5*r), t=theta[0]))
    Ecom.set_linewidth(4.0)

    ax.set_xlim((mx-dd,Mx+dd))
    ax.set_ylim((my-dd,My+dd))

    for k in range(x.size):

        Lcom.set_xdata(x[k])
        Lcom.set_ydata(y[k])
        Ex,Ey = Ellipse((x[k],y[k]), (0.5*r, r), t=theta[k])
        Ecom.set_xdata(Ex)
        Ecom.set_ydata(Ey)

        fig.canvas.draw()

  def plot(self,o=None,dt=1e-3,fign=-1,clf=True,axs0={},ls='-',ms='.',
                alpha=1.,lw=2.,fill=True,legend=True,color='k',
           plots=['2d','v'],label=None,cvt={'t':1000,'acc':1./981}):
    """
    .plot  plot trajectory

    INPUTS:
      o - Obs - trajectory to plot

    OUTPUTS:
    """
    if o is None:
      o = self.obs().resample(dt)

    t      = np.hstack(o.t) * cvt['t']
    x      = np.vstack(o.x)
    y      = np.vstack(o.y)
    theta  = np.vstack(o.theta)
    dx     = np.vstack(o.dx)
    dy     = np.vstack(o.dy)
    dtheta = np.vstack(o.dtheta)
    v      = np.vstack(o.v)
    delta  = np.vstack(o.delta)
    acc    = np.vstack(o.acc) * cvt['acc'] 

    qe      = np.vstack(o.q[::2])
    te      = np.hstack(o.t[::2]) * 1000
    xe      = np.vstack(o.x[::2])
    ye      = np.vstack(o.y[::2])
    thetae  = np.vstack(o.theta[::2])
    ve      = np.vstack(o.v)
    deltae  = np.vstack(o.delta[::2])
    thetae  = np.vstack(o.theta[::2])
    dthetae = np.vstack(o.dtheta[::2])

    fig = plt.figure(fign)
    if clf:
      plt.clf()
    axs = {}

    Np = len(plots)
    pN = 1

    if '2d' in plots:
      if '2d' in axs0.keys():
        ax = axs0['2d']
      else:
        ax = plt.subplot(Np,1,pN,aspect='equal'); pN+=1; ax.grid('on')
      #xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      #ylim = np.array([-.1,1.1])*E.mean()
      ax.plot(x ,y ,color=color,ls=ls,lw=lw,alpha=alpha,label=label)
      #ax.set_xlim(xlim); ax.set_ylim(ylim)
      ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
      axs['2d'] = ax

    if 'y' in plots:
      if 'y' in axs0.keys():
        ax = axs0['y']
      else:
        ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      #xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      #ylim = np.array([-.1,1.1])*E.mean()
      ax.plot(t,y ,color=color,ls=ls,lw=lw,alpha=alpha,label=label)
      #ax.set_xlim(xlim); ax.set_ylim(ylim)
      ax.set_ylabel('y (cm)')
      axs['y'] = ax

    if 'v' in plots:
      ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      ylim = np.array([-.1,1.1])*v.max()
      #ax.plot(np.vstack([te,te]),(np.ones((te.size,1))*ylim).T,'k:',lw=1)
      ax.plot(t,v,color=color,ls=ls,  lw=lw,label='$v$',alpha=alpha)
      ax.set_xlim(xlim); ax.set_ylim(ylim)
      if legend:
        ax.legend(loc=7,ncol=3)
      ax.set_ylabel('v (cm/sec)')
      axs['v'] = ax

    if 'acc' in plots:
      if 'acc' in axs0.keys():
        ax = axs0['acc']
      else:
        ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      ylim = np.array([min(0.,1.2*acc.min()),1.2*acc.max()])
      ax.plot(t,acc,color=color,ls=ls,  lw=lw,label='$a$',alpha=alpha)
      #ax.set_xlim(xlim); #ax.set_ylim(ylim)
      if legend:
        ax.legend(loc=7,ncol=3)
      #if fill:
      #  do_fill(te,qe,ylim)
      #ax.set_ylabel('roach perturbation (cm / s$^{-2}$)')
      ax.set_ylabel('cart acceleration (g)')
      axs['acc'] = ax

    ax.set_xlabel('time (msec)');

    return axs

  def extrinsic(self, z, q, x=0., y=0., theta=np.pi/2.):
    """
    .extrinsic  extrinsic state from intrinsic (i.e. body-centric) state

    INPUTS:
      z - 1 x 3 - (v,delta,omega) - TD state
      q - 1 x 9 - (q,m,I)

    OUTPUTS:
      x - 1 x 6 - (x,y,theta,dx,dy,dtheta)
      q - 1 x 9 - (q,m,I)
    """
    # copy data
    q = np.asarray(q).copy(); z = np.asarray(z).copy()
    # unpack params, state
    q,m,I = q[:3]
    v,delta,omega = z
    # extrinsic state variables
    dx = v*np.sin(theta + delta)
    dy = v*np.cos(theta + delta)
    dtheta = omega
    # pack params, state
    x = np.array([x,y,theta,dx,dy,dtheta])
    q = np.array([q,m,I])
    return x, q

  def intrinsic(self, x, q):
    """
    .intrinsic  body-centric state from from full state

    Inputs:
      x - 1 x 6 - (x,y,theta,dx,dy,dtheta)
      q - 1 x 9 - (q,m,I,eta0,k,d,beta,fx,fy)

    Outputs:
      z - 1 x 3 - (v,delta,omega) - TD state
      q - 1 x 9 - (q,m,I,eta0,k,d,beta,.,.)
    """
    x,y,theta,dx,dy,dtheta = x
    v = np.sqrt(dx**2 + dy**2)
    delta = np.angle(np.exp(-1j*theta)*(dy + 1j*dx))
    omega = dtheta
    z = np.array([v,delta,omega])
    return z, q

if __name__ == "__main__":

  import sys
  args = sys.argv

  op = opt.Opt()
  op.pars(fi='puck.cfg')

  p = op.p

  dt = p['dt']
  z0 = p['z0']
  q0 = p['q0']

  if 'puck' in p.keys():
    puck = p['puck']
  else:
    puck = Puck(dt=dt)

  z = z0

  X=0.; Y=0.; theta=np.pi/2.;
  #x=np.random.randn(); y=np.random.randn(); theta=2*np.pi*np.random.rand()
  x0, q0 = puck.extrinsic(z, q0, x=X, y=Y, theta=theta)
  t,x,q = puck(0, 1., x0, q0, np.inf)

  if 'plot' in args or 'anim' in args:
    o = puck.obs().resample(dt)
    if 'anim' in args:
      puck.anim(o=o)
    if 'plot' in args:
      puck.plot(o=o)

  v,delta,omega = z
  op.pars(puck=puck,
          x=X,y=Y,theta=theta,
          v=v,delta=delta,omega=omega,
          x0=x0,q0=q0,
          T=np.diff([tt[0] for tt in t[::2]]).mean())

