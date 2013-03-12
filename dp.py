
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
# (c) Humberto Gonzalez, Ram Vasudevan, UC Berkeley 2011

import numpy as np
import pylab as plt
import scipy as sp

np.set_printoptions(precision=4)

from util import Struct
from relax import HDS, Euler, stack, trans
import relax as rx

class DP(HDS):

  def __init__(self, debug=False):
    """
    dp = DP(debug)

    Double Pendulum hybrid model

    Inputs:
      debug - bool - flag for printing debugging info

    Outputs:
    dp - HDS - hybrid dynamical system
      .F - vector field   dx  = F(t,x,p) dt
      .B - stochasticity        + B(t,x,p) dw
      .R - reset map      t,x,p = R(t,x,p)
      .E - retraction     x   = E(t,x,p,v)
      .P - parameters     p   = P(q,debug)
        .j - discrete mode index
      .G - guard function g = G(t,x,p)
      .O - observations   o = O(t,x,p)
      .plot(trjs) - generate double pendulum plot from trajectory

    by Sam Burden, Humberto Gonzalez, Ram Vasudevan, Berkeley 2011
    """
    HDS.__init__(self)

  def P(self, q=[1.,1.,1.,1.,0.,0.,0.,1.,0.,1.], debug=False):
    """
    Parameters:
      q=[m1,m2,L1,L2,b1,b2,a,om,c,g]
        mi - scalar - mass of link i
        Li - scalar - length of link i
        bi - scalar - damping of joint i
        a  - scalar - torque amplitude
        om - scalar - torque frequency
        c  - scalar - coefficient of restitution
        g  - scalar - gravitational constant
      debug - bool - flag for printing debugging info
    """
    return Struct(j=None,q=q,debug=debug)

  def Lambda(self, t, x, p):
    """
    Lambda  Lagrange multiplier
    """
    m1,m2,L1,L2,b1,b2,a,om,c,g = p.q
    if p.j == 1:
      t1,t2,dt1,dt2 = x.T
    elif p.j == 2:
      t1,dt1 = x.T
      t2,dt2 = (0*t1,0*t1)

    # from Mathematica :)
    lam=(0.125E0*(L2**2*m2+L1**2*(m1+0.3E1*m2)+0.3E1*L1*L2*m2*np.cos(t2))**(-1)
         *(0.8E1*L2**2*m2*((-0.1E1)*b1*dt1+b2*dt2+a*np.sin(om*t))
         +0.4E1*dt1**2*L1**3*L2*m2*(m1+0.3E1*m2)*np.sin(t2)
         +L1**2*(0.8E1*b2*dt2*(m1+0.3E1*m2)+0.2E1*g*L2*m2*(0.6E1*m2*np.cos(t1)*np.sin(t2)
         +m1*((-0.1E1)*np.cos(t2)*np.sin(t1)+0.2E1*np.cos(t1)*np.sin(t2)))
         +0.3E1*(0.2E1*dt1**2+0.2E1*dt1*dt2+dt2**2)
         *L2**2*m2**2*np.sin(0.2E1*t2))
         +L1*L2*m2*(0.12E2*np.cos(t2)*((-0.1E1)*b1*dt1+0.2E1*b2*dt2
         +a*np.sin(om*t))+0.4E1*(dt1+dt2)**2*L2**2*m2*np.sin(t2)
         +(-0.1E1)*g*L2*(0.4E1*m1*np.sin(t1)
         +m2*(0.5E1*np.sin(t1)+(-0.3E1)*np.sin(t1+0.2E1*t2))))))

    return lam

  def G(self,t,x,p):
    """
    g = G(t,x,p)

    g > 0 : guard inactive
    g = 0 : guard 
    """
    m1,m2,L1,L2,b1,b2,a,om,c,g = p.q
    if p.j == 1:
      t1,t2,dt1,dt2 = x
      g = t2
    elif p.j == 2:
      t1,dt1 = x
      t2,dt2 = (0.,0.)
      g = self.Lambda(t,x,p)
    else:
      raise RuntimeError,"unknown discrete mode"

    return g

  def F(self,t,x,p):
    """
    dx = F(t,x,p)
    """
    m1,m2,L1,L2,b1,b2,a,om,c,g = p.q
    if p.j == 1:
      t1,t2,dt1,dt2 = x
      # from Mathematica :)
      ddt1=0.3E1*L1**(-2)*L2**(-1)*(0.8E1*m1+0.3E1*m2*(0.5E1+(-0.3E1)*np.cos(0.2E1*t2)))**(-1)*(0.12E2*b2*dt2*L1*np.cos(t2)+0.4E1*(dt1+dt2)**2*L1*L2**2*m2*np.sin(t2)+L2*(0.8E1*((-0.1E1)*b1*dt1+b2*dt2+a*np.sin(om*t))+0.3E1*dt1**2*L1**2*m2*np.sin(0.2E1*t2)+(-0.1E1)*g*L1*(0.4E1*m1*np.sin(t1)+m2*(0.5E1*np.sin(t1)+(-0.3E1)*np.sin(t1+0.2E1*t2)))));
      ddt2=0.6E1*L1**(-2)*L2**(-2)*((-0.1E1)*L2*(0.2E1*L2+0.3E1*L1*np.cos(t2))*(0.8E1*m1+0.3E1*m2*(0.5E1+(-0.3E1)*np.cos(0.2E1*t2)))**(-1)*((-0.2E1)*b1*dt1+0.2E1*a*np.sin(om*t)+L1*((-0.1E1)*g*m1*np.sin(t1)+m2*((-0.2E1)*g*np.sin(t1)+dt2*(0.2E1*dt1+dt2)*L2*np.sin(t2)))+(-0.1E1)*g*L2*m2*np.sin(t1+t2))+m2**(-1)*(L2**2*m2+L1**2*(m1+0.3E1*m2)+0.3E1*L1*L2*m2*np.cos(t2))*((-0.4E1)*m1+0.3E1*m2*((-0.4E1)+0.3E1*np.cos(t2)**2))**(-1)*(0.2E1*b2*dt2+L2*m2*(dt1**2*L1*np.sin(t2)+g*np.sin(t1+t2))));

      dx = [dt1,dt2,ddt1,ddt2]
      
    elif p.j == 2:
      t1,dt1 = x
      # from Mathematica :)
      ddt1=(-0.15E1)*(0.3E1*L1*L2*m2+L2**2*m2+L1**2*(m1+0.3E1*m2))**(-1)*(0.2E1*b1*dt1+(-0.2E1)*a*np.sin(om*t)+g*L2*m2*np.sin(t1)+L1*(g*m1*np.sin(t1)+0.2E1*g*m2*np.sin(t1)));

      dx = [dt1,ddt1]

    else:
      raise RuntimeError,"unknown discrete mode"

    return np.array(dx)

  def B(self,t,x,p):
    """
    dx = B(t,x,p)
    """
    m1,m2,L1,L2,b1,b2,a,om,c,g = p.q
    if p.j == 1:
      t1,t2,dt1,dt2 = x
      dx = [0.,0.,0.,0.]
    elif p.j == 2:
      t1,dt1 = x
      dx = [0.,0.]
    else:
      raise RuntimeError,"unknown discrete mode"

    return np.array(dx)

  def R(self,t,x,p):
    """
    t,x,p = R(t,x,p)
    """
    m1,m2,L1,L2,b1,b2,a,om,c,g = p.q
    t = t.copy()
    x = x.copy()
    p = p.copy()

    # unconstrained
    if p.j == 1:
      t1,t2,dt1,dt2 = x
      # compute constraint force
      lam = self.Lambda(t,x,p)
      # from Mathematica :)
      dt=[dt1+0.5E0*(0.1E1+c)*dt2*L2*(0.3E1*L1+0.2E1*L2)*m2*(0.3E1*L1*L2*m2+L2**2*m2+L1**2*(m1+0.3E1*m2))**(-1),(-0.1E1)*c*dt2];

      if (c == 0) and (lam > 0): # --> constrained
        x = [t1,dt[0]] 
        p.j = 2
      else: # --> unconstrained
        x = [t1,0.,dt[0],dt[1]]
        p.j = 1

    elif p.j == 2: # constrained --> unconstrained
      t1,dt1 = x
      x = [t1,0.,dt1,0.]
      p.j = 1

    else:
      raise RuntimeError,"unknown discrete mode"

    return t,np.array(x),p

  def E(self,t,x,p,v):
    """
    x = E(x,p,v)
    """
    x = x + v

    return x

  def O(self,t,x,p):
    """
    o = O(t,x,p)
    """
    m1,m2,L1,L2,b1,b2,a,om,c,g = p.q
    if p.j == 1:
      t1,t2,dt1,dt2 = x.T
    elif p.j == 2:
      t1,dt1 = x.T
      t2 = 0*t1; dt2 = 0*t1;
    else:
      raise RuntimeError,"unknown discrete mode"

    zeros = np.zeros(t1.shape)
    ones  = np.ones(zeros.shape)

    # from Mathematica :)
    T=0.166667E0*((dt1+dt2)**2*L2**2*m2+dt1**2*L1**2*(m1+0.3E1*m2)+0.3E1*dt1*(dt1+dt2)*L1*L2*m2*np.cos(t2));
    V=(-0.5E0)*g*(L1*(m1+0.2E1*m2)*np.cos(t1)+L2*m2*np.cos(t1+t2));
    L=0.166667E0*((dt1+dt2)**2*L2**2*m2+dt1**2*L1**2*(m1+0.3E1*m2)+0.3E1*dt1*(dt1+dt2)*L1*L2*m2*np.cos(t2))+0.5E0*g*(L1*(m1+0.2E1*m2)*np.cos(t1)+L2*m2*np.cos(t1+t2));

    lam = self.Lambda(t,x,p)

    F = []; B = []
    for tt,xx in zip(t,x):
      F += [self.F(tt,xx,p)]
      B += [self.B(tt,xx,p)]
    Fn = np.sqrt(np.sum(np.array(F)**2,axis=1).reshape(-1,1))
    Bn = np.sqrt(np.sum(np.array(B)**2,axis=1).reshape(-1,1))

    o = np.array((t1,t2,dt1,dt2,lam,T,V,L,Fn,Bn)).T

    return o

  def plot(self,trjs):
    """
    Plot double pendulum trajectory

    Inputs:
      trjs - list - DP trajectory

    Outputs:
      (T,X,O,P) - observations
    """
    T,X,O,P = stack(trjs)
    Te,Xe,Oe,Pe = trans(trjs)
    m1,m2,L1,L2,b1,b2,a,om,c,g = P[0].q

    fig = plt.figure(1, figsize=(6,4))
    fig.clf()

    legprop = {'size':16};
    sc = 180/np.pi
    lw = 2.; 
    ms = 20.; mew = 2.; ma = 0.5

    t = np.hstack(T)
    o = np.vstack(O)
    te = np.hstack(Te)
    oe = np.vstack(Oe)

    t1,t2,dt1,dt2,lam,TT,V,L,Fn,Bn = o.T
    t1e,t2e,dt1e,dt2e,lame,TTe,Ve,Le,Fne,Bne = oe.T

    Vm = V.min()
    V  = V - Vm
    E  = TT+V

    ax = plt.subplot(1,1,1);
    H = (ax.plot(t,sc*t1,'r',label='$\\theta_1$'),
         ax.plot(t,sc*t2,'b',label='$\\theta_2$'))
    [[hh.set_linewidth(lw) for hh in h] for h in H]
    dy = np.diff(ax.get_ylim())
    H = (ax.plot(t,0.25*dy*np.sin(om*t),label='$\\tau$',color='gray',
                 zorder=-1))
    [h.set_linewidth(2*lw) for h in H]
    H = (ax.plot(t,0.25*dy*lam/lam.max(),label='$\\lambda$',color='green',
                 zorder=-1))
    [h.set_linewidth(2*lw) for h in H]

    ylim = ax.get_ylim()
    for te,pe in zip(Te,Pe):
      if pe.j == 1:
        ax.plot([te[-1],te[-1]],ylim,'k')
      elif pe.j == 2:
        ax.plot([te[-1],te[-1]],ylim,'k--')
    M = (np.diff(1*(dt1 >= 0)) == -1).nonzero()
    for m in M:
      ax.plot([t[m],t[m]],ylim,'k:')
    ax.set_ylim(ylim)

    #H = (ax.plot(t,sc*dt1,'r-.',label='$\\dot{\\theta}_1$'),
    #     ax.plot(t,sc*dt2,'b--',label='$\\dot{\\theta}_2$'))
    #[[hh.set_linewidth(lw) for hh in h] for h in H]
    #H = (ax.plot(te,sc*t1e,'g.'),
    #     ax.plot(te,sc*t2e,'g.'))
    #for h in H:
    #  for hh in h:
    #    hh.set_ms(ms); hh.set_mew(mew); hh.set_alpha(ma)
    ax.set_xlim((t[0],t[-1]))

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('State')
    #leg = ax.legend(prop=legprop,ncol=4,loc='lower center')
    leg = ax.legend(ncol=4,loc='lower center')

    if 0:
      ax = plt.subplot(2,1,2)
      lw = 1.; al=0.1
      ms = 5.; mew = 2.; ma = 0.5

      for jj,(t,o,te,oe) in enumerate(zip(T,O,Te,Oe)):
        if jj == 0:
          continue

        t1,t2,dt1,dt2,lam,TT,V,L,Fn,Bn = o.T
        t1e,t2e,dt1e,dt2e,lame,TTe,Ve,Le,Fne,Bne = oe.T

        H = (ax.plot(t-t[0],sc*t1,'r.',label='$\\theta_1$',alpha=al),
             ax.plot(t-t[0],sc*t2,'b.',label='$\\theta_2$',alpha=al))
        [[hh.set_linewidth(lw) for hh in h] for h in H]
        [[hh.set_ms(2.) for hh in h] for h in H]
        H = (ax.plot(te-t[0],sc*t1e,'g.'),
             ax.plot(te-t[0],sc*t2e,'g.'))
        for h in H:
          for hh in h:
            hh.set_ms(ms); hh.set_mew(mew); hh.set_alpha(ma)

      ax.set_xlabel('Time (sec)')
      ax.set_ylabel('State')

    #plt.show()
    return T,X,O,P


if __name__ == "__main__":

  m1 = 1.; m2 = 1.; L1 = 1.; L2 = 1.;
  #b1 = 0.; b2 = 0.; a = 0.; om = 1.;
  b1 = 0.3; b2 = 0.3; a = 0.5; om = 2*np.pi/32
  c = 0.; g = 1.;
  q = [m1,m2,L1,L2,b1,b2,a,om,c,g]
  debug = False

  dp = DP()
  p = dp.P(q,debug)

  #tf = np.infty
  #n  = 40
  dt = 1.
  t0 = (0.5 + dt)*np.pi/om 
  tf = (6.*2 + 0.5 + dt)*np.pi/om 
  n  = np.infty
  t = (t0,tf)
  j = 1
  x = np.array([-np.pi/4, np.pi/8, 0, 0])
  #j = 2
  #x = np.array([0.26,-0.01])
  #x = np.array([np.pi/10, 0.1*np.random.randn()])
  p.j = j

  h  = 5e-2
  eps = 2e-1
  Zeno = False

  import cProfile
  import time; start = time.clock()
  trjs = Euler(t, x, p, dp, h, eps, n, debug, Zeno)
  #cProfile.run('trjs = Euler(t, x, p, dp, h, eps, n, debug, Zeno)')
  print "%0.2f sec" % (time.clock() - start)

  dp.plot(trjs)

  def flow(dp,p,h,eps,ph,x):
    """
    trjs = flow  compute flow between two phases
    """
    m1,m2,L1,L2,b1,b2,a,om,c,g = p.q

    t0 = ph[0] / om; tf = ph[-1] / om; n = np.inf

    x = np.array(x)
    # don't simulate from invalid initial condition
    if p.j == 2 and dp.Lambda(t0,x,p) <= 0:
      return np.nan*x

    trjs = rx.Euler((t0,tf), x, p, dp, h, eps, n)

    if len(trjs[-1].t) <= 2:
      trjs = trjs[:-1]
    trj = trjs[-1]

    s = (tf - trj.t[-2]) / (trj.t[-1]-trj.t[-2])
    trj.t[-1] = (1-s)*trj.t[-2] + s*trj.t[-1]
    trj.x[-1] = (1-s)*trj.x[-2] + s*trj.x[-1]

    return trjs

  # Deformation of Poincare section
  psi = (lambda t,x : flow(dp,p,h,eps,((0.5+dt)*np.pi,(0.5+dt)*np.pi+2*np.pi*t),x)[-1].x[-1])

  # Poincare map
  pmap = lambda x : psi(1.,x)

  # find periodic orbit
  x0 = rx.fp(pmap,x)
  # linearize Poincare map
  J = rx.jac(pmap,x0)
  # compute eigenvalues of P-map
  print np.abs(np.linalg.eigvals(J))
  
