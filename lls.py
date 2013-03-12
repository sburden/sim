
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
	
class LLS(hds.HDS):
  def __init__(self,dt=1./500):
    """
    LLS  LLS hybrid system
    """
    super(LLS, self).__init__(dt=dt)
    
    self.name = 'LLS'
    self.accel = lambda t,x,q : np.zeros((x.shape[0],3))
      
  def P(self, q=[1,0.0025,2.04e-7,0.017,0.53,-0.002,np.pi/4,0.,0.], debug=False):
    """
    Parameters:
      q=[m,M,k,b,l0,a,om,ph,g]
         q = 1
         m = 0.0025
         I = 2.04e-7
         eta0 = 0.017
         k = 0.53
         d = -0.002
         beta = np.pi/4
         fx=0.
         fy=0.
      debug - bool - flag for printing debugging info
    """
    return Struct(j=q[0],q=q,debug=debug)

  def dVdeta(self, eta, q):
    """
    dVdeta  spring force
    """
    # unpack params
    q,m,I,eta0,k,d,beta,fx,fy = q
    # linear spring
    return k*(eta - eta0)

  def dyn(self, t, x, q):
    """
    .dyn  evaluates system dynamics
    """
    p = q.copy()
    # perturbation
    acc = self.accel(t,np.array([x]),q).flatten()
    # unpack params
    q,m,I,eta0,k,d,beta,fx,fy = q
    # unpack state
    x,y,theta,dx,dy,dtheta = x
    # foot, COM, hip
    f = np.array([fx,fy])
    c = np.array([x,y])
    h = c + d*np.array([np.sin(theta),np.cos(theta)])
    # leg length
    eta = np.linalg.norm(h - f)
    # spring force
    dV = self.dVdeta(eta,p)
    # Cartesian dynamics
    dx = [dx, dy, dtheta,
          -dV*(x + d*np.sin(theta) - fx)/(m*eta) + acc[0],
          -dV*(y + d*np.cos(theta) - fy)/(m*eta) + acc[1],
          -dV*d*((x - fx)*np.cos(theta) 
               - (y - fy)*np.sin(theta))/(I*eta) + acc[2]]
    # return vector field
    return dx

  def obs(self):
    """
    .obs  observes trajectory
    """
    o = hds.Obs(t=self.t,x=[],y=[],theta=[],dx=[],dy=[],dtheta=[],
                         fx=[],fy=[],PE=[],KE=[],E=[],v=[],delta=[],
                         q=[],hx=[],hy=[],acc=[])

    for t,x,q in zip(self.t,self.x,self.q):
      # perturbation
      acc = self.accel(t,x,q)
      # unpack params
      q,m,I,eta0,k,d,beta,fx,fy = q
      # pre-allocate
      ones = np.ones(x.shape[0])
      # unpack state
      x,y,theta,dx,dy,dtheta = x.T
      # foot
      fx = fx*ones
      fy = fy*ones
      # discrete mode 
      q = q*ones
      # foot, COM, hip
      f = np.array([fx,fy])
      c = np.array([x,y])
      h = c + d*np.array([np.sin(theta),np.cos(theta)])
      # leg length
      eta = np.sqrt(np.sum((h - f)**2,axis=0))
      # energies
      PE = (k/2)*(eta - eta0)**2
      KE = m*(dx**2 + dy**2)/2 + I*(dtheta**2)/2
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
      o.fx += [np.c_[fx]]
      o.fy += [np.c_[fy]]
      o.hx += [np.c_[h[0]]]
      o.hy += [np.c_[h[1]]]
      o.PE += [np.c_[PE]]
      o.KE += [np.c_[KE]]
      o.E += [np.c_[PE+KE]]
      o.v += [np.c_[v]]
      o.delta += [np.c_[delta]]
      o.q += [np.c_[q]]
      o.acc += [np.c_[acc]]
    # store result
    self.o = o
    return self.o

  def trans(self, t, x, q, e):
    """
    .trans  transition between discrete modes
    """
    # copy data
    t = t.copy(); x = x.copy(); q = q.copy()
    # unpack params
    q,m,I,eta0,k,d,beta,fx,fy = q
    # unpack state
    x,y,theta,dx,dy,dtheta = x
    # foot, com, hip
    f = np.array([fx,fy])
    c = np.array([x,y])
    h = c + d*np.array([np.sin(theta),np.cos(theta)])
    # leg length
    eta = np.linalg.norm(h - f)
    # foot in body frame
    fh = f-h
    fb = [fh[0]*np.cos(-theta)  + fh[1]*np.sin(-theta),
          fh[0]*-np.sin(-theta) + fh[1]*np.cos(-theta)]
    # left foot is right of body axis OR right foot is left of body axis
    if ( ( fb[0] > 0 ) and ( q == 0 ) ) or ( ( fb[0] < 0 ) and ( q == 1 ) ):
      q=-1
    else:	
      # switch stance foot
      q = np.mod(q + 1,2)
      # COM, hip
      c = np.array([x,y])
      h = c + d*np.array([np.sin(theta),np.cos(theta)])
      # leg reset; q even for left stance, odd for right
      eta = eta0*np.array([np.sin(theta - beta*(-1)**q),
                           np.cos(theta - beta*(-1)**q)])
      # new foot position
      f = h + eta
      fx = f[0]
      fy = f[1]
      # com vel
      dc = np.array([dx,dy])
      # hip vel
      dh = dc + d*dtheta*np.array([np.cos(theta),-np.sin(theta)])
      # hip vel in body frame 
      dhb=[dh[0]*np.cos(-theta)  + dh[1]*np.sin(-theta),
           dh[0]*-np.sin(-theta) + dh[1]*np.cos(-theta)]
      # foot in body frame
      fh=f-h
      fb=[fh[0]*np.cos(-theta)  + fh[1]*np.sin(-theta),
          fh[0]*-np.sin(-theta) + fh[1]*np.cos(-theta)]
      # leg will instantaneously extend 
      if np.dot(fb, dhb) < 0:
        q=-1
    # pack state, params
    x = np.array([x,y,theta,dx,dy,dtheta])
    q = np.array([q,m,I,eta0,k,d,beta,fx,fy])
    return t, x, q

  def evts(self, q):
    """
    .evts  returns event functions for given hybrid domain
    """
    # unpack params
    q,m,I,eta0,k,d,beta,fx,fy = q
    # leg extension
    def leg(t,x):
      # unpack state
      x,y,theta,dx,dy,dtheta = x
      # foot, COM, hip
      f = np.array([fx,fy])
      c = np.array([x,y])
      h = c + d*np.array([np.sin(theta),np.cos(theta)])
      # leg length
      eta = np.linalg.norm(h - f)
      # leg extension
      return eta - eta0
    # foot location
    def foot(t,x):
      # unpack state
      x,y,theta,dx,dy,dtheta = x
      # foot, COM, hip
      f = np.array([fx,fy])
      c = np.array([x,y])
      h = c + d*np.array([np.sin(theta),np.cos(theta)])
      # foot in body frame
      fh = f-h
      fb = [fh[0]*np.cos(-theta)  + fh[1]*np.sin(-theta),
            fh[0]*-np.sin(-theta) + fh[1]*np.cos(-theta)]
      # foot distance from body axis
      return fb[0]*(-1)**q
    # leg extension, foot location
    return [leg, foot]

  def phi(self,tf,x0,q0,t0=0.,debug=False):
    """
    t,x,q = phi(t,x0,q0)  hybrid flow

    Inputs:
      tf - scalar - final time
      x0 - initial state
      q0 - initial params
      (optional)
      t0 - scalar - initial time

    Outputs:
      T - times
      X - states
      Q - params

    """
    if tf == t0:
      T = [np.array([t0])]
      X = [np.array(x0)]
      Q = [np.array(q0)]
      
    else:
      self(t0,tf,x0,q0,np.inf,clean=True)

      T = np.hstack(self.t)
      X = np.vstack(self.x)
      Q = np.vstack([np.ones((x.shape[0],1))*q for x,q in zip(self.x,self.q)])

    return T,X,Q

  def step(self, z0, q0, steps=2):
    """
    .step  LLS stride from touchdown in body-centric coords

    Inputs:
      z - 1 x 3 - (v,delta,omega)
      q - 1 x 9 - (q,m,I,eta0,k,d,beta,.,.)
      (optional)
      steps - int - number of LLS steps to take

    Outputs:
      z - 1 x 3 - (v,delta,omega)
    """
    # instantiate extrinsic coords
    x0,q0 = self.extrinsic(z0, q0)
    # simulate for specified number of steps
    lls = self; lls.__init__(dt=self.dt)
    t,x,q = lls(0, 1e99, x0, q0, steps)
    # extract intrinsic coords
    z,_ = self.intrinsic(x[-1][-1], q[-1])
    return z

  def omap(self, z, args):
    """
    .omap  LLS orbit map in (v,delta) coordinates

    INPUTS
      z - 1 x 3 - (v,delta,omega)
        v - speed
        delta - heading
        omega - angular velocity
      args - (q)
        q - 1 x 9 - (q,m,I,eta0,k,d,beta,.,.)

    OUTPUTS
      z - 1 x 3 - (v,delta,omega)
    """
    if np.isnan(z).any():
        return np.nan*z
    q, = args

    return self.step(z, q)

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
    fx = np.vstack(o.fx)
    fy = np.vstack(o.fy)
    v = np.vstack(o.v)
    delta = np.vstack(o.delta)
    theta = np.vstack(o.theta)
    dtheta = np.vstack(o.dtheta)
    PE = np.vstack(o.PE)
    KE = np.vstack(o.KE)
    E = np.vstack(o.E)

    te = np.hstack(o.t[::2])
    xe = np.vstack(o.x[::2])
    ye = np.vstack(o.y[::2])
    thetae = np.vstack(o.theta[::2])
 
    z = np.array([v[-1],delta[-1],theta[-1],dtheta[-1]])

    def zigzag(a=.2,b=.6,c=.2,p=4,N=100):
      x = np.linspace(0.,a+b+c,N); y = 0.*x
      mb = np.round(N*a/(a+b+c)); Mb = np.round(N*(a+b)/(a+b+c))
      y[mb:Mb] = np.mod(np.linspace(0.,p-.01,Mb-mb),1.)-0.5
      return np.vstack((x,y))

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
    Lft,  = ax.plot([x[0],fx[0]],[y[0],fy[0]],'g.-',lw=4.)

    ax.set_xlim((mx-dd,Mx+dd))
    ax.set_ylim((my-dd,My+dd))

    for k in range(x.size):

        Lcom.set_xdata(x[k])
        Lcom.set_ydata(y[k])
        Lft.set_xdata([x[k],fx[k]])
        Lft.set_ydata([y[k],fy[k]])
        Ex,Ey = Ellipse((x[k],y[k]), (0.5*r, r), t=theta[k])
        Ecom.set_xdata(Ex)
        Ecom.set_ydata(Ey)

        fig.canvas.draw()

  def plot(self,o=None,dt=1e-3,fign=-1,clf=True,axs0={},ls='-',ms='.',
                alpha=1.,lw=2.,fill=True,legend=True,color='k',
           plots=['2d','v','E'],label=None,cvt={'t':1000,'acc':1./981}):
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
    fx     = np.vstack(o.fx)
    fy     = np.vstack(o.fy)
    PE     = np.vstack(o.PE)
    KE     = np.vstack(o.KE)
    E      = np.vstack(o.E)
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
    fxe    = np.vstack(o.fx[::2])
    fye    = np.vstack(o.fy[::2])

    def do_fill(te,qe,ylim):
      for k in range(len(te)-1):
        if qe[k,0] == 0:
          color = np.array([1.,0.,0.])
        if qe[k,0] == 1:
          color = np.array([0.,0.,1.])
        ax.fill([te[k],te[k],te[k+1],te[k+1]],
                [ylim[1],ylim[0],ylim[0],ylim[1]],
                fc=color,alpha=.35,ec='none',zorder=-1)
 
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
      if fill:
        #ax.plot(xe,ye,'y+',mew=2.,ms=8)
        ax.plot(fxe[qe==0.],fye[qe==0.],'ro',mew=0.,ms=10)
        ax.plot(fxe[qe==1.],fye[qe==1.],'bo',mew=0.,ms=10)
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
      if fill:
        do_fill(te,qe,ylim)
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

    if 'E' in plots:
      ax = plt.subplot(Np,1,pN); pN+=1; ax.grid('on')
      xlim = np.array([te[0],te[-1]]) + np.array([-.02,.02])*(te[-1]-te[0])
      ylim = np.array([-.1,1.1])*E.mean()
      ax.plot(t,E ,color=color,ls=ls,lw=lw,label='$E$',alpha=alpha)
      ax.plot(t,KE,'b',ls=ls,  lw=lw,label='$KE$',alpha=alpha)
      ax.plot(t,PE,'g',ls=ls,  lw=lw,label='$PE$',alpha=alpha)
      ax.set_xlim(xlim); ax.set_ylim(ylim)
      if legend:
        ax.legend(loc=7,ncol=3)
      for k in range(len(te)-1):
        if qe[k,0] == 0 and fill:
          ax.fill([te[k],te[k],te[k+1],te[k+1]],
                  [ylim[1],ylim[0],ylim[0],ylim[1]],
                  fc=np.array([1.,1.,1.])*0.75,ec='none',zorder=-1)
      ax.set_ylabel('E (g m**2 / s**2)')
      axs['E'] = ax

    ax.set_xlabel('time (msec)');

    return axs

  def extrinsic(self, z, q, x=0., y=0., theta=np.pi/2.):
    """
    .extrinsic  extrinsic LLS state from intrinsic (i.e. Poincare map) state

    INPUTS:
      z - 1 x 3 - (v,delta,omega) - TD state
      q - 1 x 9 - (q,m,I,eta0,k,d,beta,.,.)

    OUTPUTS:
      x - 1 x 6 - (x,y,theta,dx,dy,dtheta)
      q - 1 x 9 - (q,m,I,eta0,k,d,beta,fx,fy)
    """
    # copy data
    q = np.asarray(q).copy(); z = np.asarray(z).copy()
    # unpack params, state
    q,m,I,eta0,k,d,beta,_,_ = q
    v,delta,omega = z
    # extrinsic state variables
    dx = v*np.sin(theta + delta)
    dy = v*np.cos(theta + delta)
    dtheta = omega
    # COM, hip
    c = np.array([x,y])
    h = c + d*np.array([np.sin(theta),np.cos(theta)])
    # foot
    f = h + eta0*np.array([np.sin(theta - beta*(-1)**q),
                           np.cos(theta - beta*(-1)**q)])
    fx = f[0]; fy = f[1]
    # pack params, state
    x = np.array([x,y,theta,dx,dy,dtheta])
    q = np.array([q,m,I,eta0,k,d,beta,fx,fy])
    return x, q

  def intrinsic(self, x, q):
    """
    .intrinsic  Poincare map state from from full state

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
  op.pars(fi='lls.cfg')
  #op.pars(fi='dbg.cfg')

  p = op.p

  dt = p['dt']
  z0 = p['z0']
  q0 = p['q0']
  N  = p['N']
  dbg  = p.get('dbg',False)

  if 'lls' in p.keys():
    lls = p['lls']
  else:
    lls = LLS(dt=dt)

  st = time.time()
  z = lls.ofind(np.asarray(z0),(q0,),N=10,modes=[1])
  print '%0.2fsec to find gait, z = %s' % (time.time() - st,z)

  X=0.; Y=0.; theta=np.pi/2.;
  #x=np.random.randn(); y=np.random.randn(); theta=2*np.pi*np.random.rand()
  x0, q0 = lls.extrinsic(z, q0, x=X, y=Y, theta=theta)
  t,x,q = lls(0, 1e99, x0, q0, N, dbg=dbg)

  if 'plot' in args or 'anim' in args:
    o = lls.obs().resample(dt)
    if 'anim' in args:
      lls.anim(o=o)
    if 'plot' in args:
      lls.plot(o=o)

  v,delta,omega = z
  op.pars(lls=lls,
          x=X,y=Y,theta=theta,
          v=v,delta=delta,omega=omega,
          x0=x0,q0=q0,
          T=np.diff([tt[0] for tt in t[::2]]).mean())

