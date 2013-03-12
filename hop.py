
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
import scipy as sp
import scipy.optimize as op

from IPython.Debugger import Tracer; dbg = Tracer()

np.set_printoptions(precision=4)

from util import Struct
import relax as rx

euler = rx.Euler

class Hop(rx.HDS):

  def __init__(self):
    """
    hop = Hop()

    hop - HDS - Vertical Hopper hybrid model
      .F - vector field   dx  = F(t,x,p) dt
      .B - stochasticity        + B(t,x,p) dw
      .R - reset map      t,x,p = R(t,x,p)
      .E - retraction     x   = E(t,x,p,v)
      .P - parameters     p   = P(q,debug)
        .j - discrete mode index
      .G - guard function g = G(t,x,p)
      .O - observations   o = O(t,x,p)
      .plot(trjs) - generate  plot from trajectory

    by Sam Burden, Shai Revzen, Berkeley 2011
    """
    rx.HDS.__init__(self)

  def P(self, q=[1.,2.,10.,0.1,1.,0.25,2*np.pi/2.,0.,1.], debug=False):
    """
    Parameters:
      q=[m,M,k,b,l0,a,om,ph,g]
        m,M - masses
        k,b,l0 - spring constant, damping coefficient, nominal length
        a,om,ph - leg actuation magnitude, frequency, phase offset
        g - gravitational constant
      debug - bool - flag for printing debugging info
    """
    return Struct(j=None,q=q,debug=debug)

  def Leg(self, t, z, p):
    """
    Leg  vertical leg force
    """
    m,M,k,b,l0,a,om,ph,g = p.q
    if p.j == 1:
      x,y,dx,dy = z.T
    elif p.j == 2:
      y,dy = z.T
      x,dx = (0.*y,0.*dy)
    else:
      raise RuntimeError,"unknown discrete mode"

    leg = k*((l0 + a*np.sin(om*t + ph)) - (y - x))

    return leg

  def G(self,t,z,p):
    """
    g = G(t,z,p)

    g > 0 : guard inactive
    g = 0 : guard 
    """
    m,M,k,b,l0,a,om,ph,g = p.q
    if p.j == 1:
      x,y,dx,dy = z.T
      g = x # height of foot
    elif p.j == 2:
      y,dy = z.T
      x,dx = (0.*y,0.*dy)
      g = g*m + self.Leg(t,z,p) # ground reaction force
    else:
      raise RuntimeError,"unknown discrete mode"

    return g

  def F(self,t,z,p):
    """
    dz = F(t,z,p)
    """
    m,M,k,b,l0,a,om,ph,g = p.q
    if p.j == 1:
      x,y,dx,dy = z.T
      leg = self.Leg(t,z,p)
      ddx = (-g*m - leg - b*dx) / m
      ddy = (-g*M + leg) / M
      dz = [dx,dy,ddx,ddy]
    elif p.j == 2:
      y,dy = z.T
      leg = self.Leg(t,z,p)
      ddy = (-g*M + leg) / M
      dz = [dy,ddy]
    else:
      raise RuntimeError,"unknown discrete mode"

    return np.array(dz)

  def B(self,t,z,p):
    """
    dx = B(t,z,p)
    """
    m,M,k,b,l0,a,om,ph,g = p.q
    if p.j == 1:
      x,y,dx,dy = z.T
      leg = self.Leg(t,z,p)
      dz = [0.,0.,0.,0.]
    elif p.j == 2:
      y,dy = z.T
      leg = self.Leg(t,z,p)
      dz = [0.,0.]
    else:
      raise RuntimeError,"unknown discrete mode"

    return np.array(dz)

  def R(self,t,z,p):
    """
    t,z,p = R(t,z,p)
    """
    m,M,k,b,l0,a,om,ph,g = p.q
    t = t.copy()
    z = z.copy()
    p = p.copy()

    # unconstrained
    if p.j == 1:
      x,y,dx,dy = z.T
      # compute constraint force
      leg = self.Leg(t,z,p)
      if g*m + leg > 0: # --> constrained
        z = [y,dy] 
        p.j = 2
      else: # --> unconstrained
        z = [0.,y,0.,dy]
        p.j = 1

    elif p.j == 2: # constrained --> unconstrained
      y,dy = z.T
      z = [0.,y,0.,dy]
      p.j = 1

    else:
      raise RuntimeError,"unknown discrete mode"

    return t,np.array(z),p

  def E(self,t,z,p,v):
    """
    z = E(z,p,v)
    """
    z = z + v

    return z

  def O(self,t,z,p):
    """
    o = O(t,z,p)
    """
    m,M,k,b,l0,a,om,ph,g = p.q
    if p.j == 1:
      x,y,dx,dy = z.T
    elif p.j == 2:
      y,dy = z.T
      x,dx = (0.*y,0.*dy)
    else:
      raise RuntimeError,"unknown discrete mode"

    zeros = np.zeros(x.shape)
    ones  = np.ones(zeros.shape)

    T = 0.5*m*dx**2 + 0.5*M*dy**2
    V = m*g*x + m*g*y
    L = T+V

    leg = self.Leg(t,z,p)

    F = []; B = []
    for tt,zz in zip(t,z):
      F += [self.F(tt,zz,p)]
      B += [self.B(tt,zz,p)]
    Fn = np.sqrt(np.sum(np.array(F)**2,axis=1).reshape(-1,1))
    Bn = np.sqrt(np.sum(np.array(B)**2,axis=1).reshape(-1,1))

    o = np.array((x,y,dx,dy,leg,T,V,L,Fn,Bn)).T

    return o

  def Y(self,trjs,N):
    """
    y = Y(t,o)
    """
    T,Z,O,P = rx.stack(trjs)

    t = np.hstack(T)
    o = np.vstack(O)

    x,y,dx,dy,leg,TT,V,L,Fn,Bn = o.T

    tp = t; yp = y; dyp = dy
    t = np.linspace(t[0],t[-1],N,endpoint=False)
    y = np.interp(t,tp,yp,left=np.nan,right=np.nan)
    dy = np.interp(t,tp,dyp,left=np.nan,right=np.nan)

    return t,np.array((y)).T
    #return t,np.array((y,dy)).T

  def plot(self,trjs,fn='',dpi=None,pxh=None,pxw=None,
                     alpha=1.,lw=4,clf=True):
    """
    Plot trajectory

    Inputs:
      trjs - list - Hop trajectory

    Outputs:
      (T,Z,O,P) - observations
    """
    T,Z,O,P = rx.stack(trjs)
    Te,Ze,Oe,Pe = rx.trans(trjs)
    m,M,k,b,l0,a,om,ph,g = P[0].q

    fsz = (8.,4.)
    if pxh:
      dpi = fsz(1)/pxh
    elif pxw:
      dpi = fsz(0)/pxw


    fig = plt.figure(1,figsize=fsz)
    #fig = plt.figure(1)
    if clf:
      fig.clf()

    legprop = {'size':10};
    sc = 1.
    ms = 20.; mew = 2.; ma = 0.5

    t = np.hstack(T)
    o = np.vstack(O)
    te = np.hstack(Te)
    oe = np.vstack(Oe)

    x,y,dx,dy,leg,TT,V,L,Fn,Bn = o.T
    xe,ye,dxe,dye,lege,TTe,Ve,Le,Fne,Bne = oe.T

    Vm = V.min()
    V  = V - Vm
    E  = TT+V

    ax = plt.subplot(2,1,1); plt.grid('on')
    H = (ax.plot(t,sc*x,'r',label='$x(t)$',alpha=alpha),
         ax.plot(t,sc*y,'b',label='$y(t)$',alpha=alpha))
    [[hh.set_linewidth(lw) for hh in h] for h in H]
    #H = (ax.plot(t,0.5*np.sin(om*t+ph),label='$f(t) \\propto \\sin(\\omega t+\pi)$',color='k',zorder=-1,lw=2.,alpha=alpha))

    ylim = np.array(ax.get_ylim())#-0.25
    ylim = [-1.5,3.0]
    for te,pe in zip(Te,Pe):
      if pe.j == 2:
        ax.fill([te[0],te[0],te[1],te[1]],
                [ylim[1],ylim[0],ylim[0],ylim[1]],
                fc=np.array([1.,1.,1.])*0.75,ec='none',zorder=-1)
    ax.set_ylim(ylim)
    ax.set_ylabel('height')

    #leg = ax.legend(ncol=4,loc='lower center')
    if clf:
      leg = ax.legend(ncol=4,loc='lower left',prop=legprop)

    if clf:
      ax.set_xlim((t[0],t[-1]))

    ax = plt.subplot(2,1,2); plt.grid('on')
    H = (ax.plot(t,sc*dx,'r',label='$\\dot{x}(t)$',alpha=alpha),
         ax.plot(t,sc*dy,'b',label='$\\dot{y}(t)$',alpha=alpha))
    [[hh.set_linewidth(lw) for hh in h] for h in H]
    #H = (ax.plot(t,0.5*np.sin(om*t+ph),label='$f(t) \\propto \\sin(\\omega t+\pi)$',color='k',zorder=-1,lw=2.,alpha=alpha))

    ylim = np.array(ax.get_ylim())#-0.25
    #ylim = [-1.5,3.0]
    for te,pe in zip(Te,Pe):
      if pe.j == 2:
        ax.fill([te[0],te[0],te[1],te[1]],
                [ylim[1],ylim[0],ylim[0],ylim[1]],
                fc=np.array([1.,1.,1.])*0.75,ec='none',zorder=-1)
    ax.set_ylim(ylim)
    ax.set_xlabel('time (t)')
    ax.set_ylabel('velocity')

    if clf:
      leg = ax.legend(ncol=4,loc='lower left',prop=legprop)

    # nominal spring length
    #ax.plot([t[0],t[-1]],[l0,l0],'k--',lw=2)

    plt.subplot(2,1,1)

    if clf:
      ax.set_xlim((t[0],t[-1]))

    if not fn == '':
      plt.savefig(fn,bbox_inches='tight',pad_inches=0.1,dpi=dpi)
    
    #plt.show()
    return T,Z,O,P

  def sch(self,t,z,p,figN=2,text=False,fn='',dpi=None,pxh=None,pxw=None):
    """
    Plot schematic

    Inputs:
      t,z,p - time, state, parameters
    """

    def zigzag(a=.2,b=.6,c=.2,p=4,N=100):
      x = np.linspace(0.,a+b+c,N); y = 0.*x
      mb = np.round(N*a/(a+b+c)); Mb = np.round(N*(a+b)/(a+b+c))
      y[mb:Mb] = np.mod(np.linspace(0.,p-.01,Mb-mb),1.)-0.5
      return np.vstack((x,y))

    m,M,k,b,l0,a,om,ph,g = p.q
    o = self.O(np.array([t]),np.array([z]),p)
    x,y,dx,dy,leg,TT,V,L,Fn,Bn = o.T

    fsz = (8.,6.)
    if pxh:
      dpi = pxh/fsz(1)
    elif pxw:
      dpi = pxw/fsz(0)


    fig = plt.figure(figN); fig.clf()
    #ax = plt.axes([0.5,0.,0.5,1.])
    if not text:
      ax = plt.axes([0.0,0.,0.35,1.])
    else:
      ax = plt.axes([0.0,0.,0.35,0.60])

    sc = 1.
    lw = 4.; 
    ms = 20.; mew = 2.; ma = 0.5
    fs = 20.
    fd = {'family':'serif','size':fs,
          'weight':'bold'}

    xlim = (-0.75,0.75)
    ylim = (-0.25,2*l0)

    sc = 1./1.5
    mw = 0.5*sc; Mw = 1.*sc
    mh = 1.*mw; Mh = 1.*Mw

    gc = np.array([139.,69.,19.])/255.
    mc = np.array([1.0,0.4,0.4])
    Mc = np.array([0.4,0.4,1.])
    lc = np.array([0.,0.5,0.])

    # rail
    #ax.plot([0.,0.],ylim,'--',lw=lw,zorder=-1,color='gray')
    # m
    ax.fill([-mw,-mw,mw,mw],[x+mh,x,x,x+mh],lw=lw,ec=0.5*mc,fc=mc)
    # M
    ax.fill([-Mw,-Mw,Mw,Mw],[y+Mh,y,y,y+Mh],lw=lw,ec=0.5*Mc,fc=Mc)
    # spring
    ly,lx = zigzag()
    ly = (y-x-mh)*ly+x+mh; lx = .25*lx + 0.2;
    #ax.plot([0.,0.],[x+mh/2.,y+Mh/2.],lw=3*lw,color=lc,zorder=-1)
    ax.plot(lx,ly,lw=1.5*lw,color=lc,zorder=-1)
    # actuator
    phi = np.sin(om*t + ph)
    ax.arrow(-0.65*mw,(x+mh+y)/2.,0.,0.2*phi*(y-x-mh)[0],head_width=0.125,lw=lw,fc=Mc,ec=Mc)
    ax.arrow(-0.65*mw,(x+mh+y)/2.,0.,-0.2*phi*(y-x-mh)[0],head_width=0.125,lw=lw,fc=mc,ec=mc)
    if text:
      xlim = (-1.2,1.2)
      ylim = (-0.25,1.5*l0)
      # m
      ax.text(0.,x+mh/2,'$m, b$',ha='center',va='center',**fd)
      # x
      ax.text(mw+mw/2,x+0.05,'$x(t)$',ha='left',va='bottom',**fd)
      ax.plot([1.8*mw+0.25*mw,1.8*mw+0.25*mw],[0.,x[0]],'--',lw=lw,color='gray')
      ax.plot([1.8*mw,1.8*mw+0.5*mw],[x,x],lw=lw,color='gray')
      # M
      ax.text(0.,y+Mh/2,'$\\mu$',ha='center',va='center',**fd)
      # y
      ax.text(Mw+mw/2-0.1,y+0.05,'$y(t)$',ha='left',va='bottom',**fd)
      #ax.plot([1.8*mw+Mw-mw+0.25*mw,1.8*mw+Mw-mw+0.25*mw],[x[0]+mh,y[0]],'--',lw=lw,color='gray')
      ax.plot([1.8*mw+Mw-mw+0.25*mw,1.8*mw+Mw-mw+0.25*mw],[0,y[0]],'--',lw=lw,color='gray')
      ax.plot([1.8*mw+Mw-mw,1.8*mw+Mw-mw+0.5*mw],[y,y],lw=lw,color='gray')
      #ax.plot([1.8*mw+Mw-mw,1.8*mw+Mw-mw+0.5*mw],[x+mh,x+mh],lw=lw,color='gray')
      # leg spring
      ax.text(mw/1.+0.1,(x+mh+y)/2.,'$k, \\ell_0$',va='center',**fd)
      # leg actuator
      #ax.arrow(-1.*mw,mh/4+(x+mh+y)/2.+Mh/4.,0.,-0.65*Mh,head_width=0.125,lw=lw,fc='k')
      #ax.arrow(-0.65*mw,(x+mh+y)/2.+Mh/4.-0.75*Mh,0.,0.85*Mh,head_width=0.125,lw=lw,fc='k')
      ax.text(-0.85*mw,(x+mh+y)/2.,'$f(t)$',ha='right',va='center',**fd)
      # g
      ax.arrow(-2.75*mw,mh+Mh+(x+mh+y)/2.+Mh/2.,0.,-Mh,
               head_width=0.125,lw=lw,fc='k')
      ax.text( -3.0*mw,mh+Mh+(x+mh+y)/2.,'$g$',ha='right',va='center',**fd)

    # ground
    ax.fill([2*xlim[0],2*xlim[0],2*xlim[1],2*xlim[1]],
             [0.,ylim[0],ylim[0],0.],lw=lw,ec=0.5*gc,fc=gc,zorder=5)

    ax.axis('equal')
    ax.set_xlim(xlim); ax.set_xticks([])
    ax.set_ylim(ylim); ax.set_yticks([])

    #plt.show()

    if not fn == '':
      plt.savefig(fn,bbox_inches='tight',pad_inches=-0.1,dpi=dpi)
      #plt.savefig(fn)

def flow(hop,p,h,eps,t,x,debug=False):
  """
  trjs = flow  compute flow between two phases
  """
  m,M,k,b,l0,a,om,ph0,g = p.q

  t0 = t[0]; tf = t[-1]; n = np.inf

  x = np.array(x)
  # don't simulate from invalid initial condition
  if p.j == 2:
    assert( -hop.Leg( t0,x,p ) - m*g <= 0 ) 

  trjs = euler((t0,tf), x, p, hop, h, eps, n, debug=debug)

  # clean up relaxed states,times
  if len(trjs[-1].t) <= 2:
    trjs = trjs[:-1]
  trj = trjs[-1]

  # terminate exactly at final time
  s = (tf - trj.t[-2]) / (trj.t[-1]-trj.t[-2])
  trj.t[-1] = (1-s)*trj.t[-2] + s*trj.t[-1]
  trj.x[-1] = (1-s)*trj.x[-2] + s*trj.x[-1]

  return trjs

def do_fp(h=1e-3,eps=1e-10,debug=False):
  q=[1.,2.,10.,5.,2.,2.0,2*np.pi,0,2.]
  m,M,k,b,l0,a,om,ph,g = q
  debug = False

  hop = Hop()
  p = hop.P(q,debug)

  ph0 = np.pi# - np.pi/(2**2)
  phf = ph0 + 20*2*np.pi
  t0 = 0.; tf = (phf-ph0)/om
  n  = np.infty
  t = (t0,tf)
  j = 2
  z = np.array([2.0,2.0])
  #j = 1
  #z = np.array([0.1,2.0,0.,2.0])
  ph = ph0
  p.j = j
  q = [m,M,k,b,l0,a,om,ph,g]
  p.q = q

  Zeno = False

  import time; 
  
  start = time.clock()
  trjs = flow(hop, p, h, eps, (t0,tf), z, debug=debug)

  #hop.plot(trjs[-10:])
  hop.plot(trjs)
  xlim = plt.xlim(); ylim = plt.ylim()
  plt.xlim(xlim); plt.ylim(ylim)

  print "%0.2f sec for sim & plot" % (time.clock() - start)

  #dbg()

  qq = q[:]
  pp = hop.P(qq); pp.j = trjs[-1].p.j
  zz = trjs[-1].x[-1]
  # Deformation of Poincare section
  psi = (lambda t,x : flow(hop,pp,h,eps,(0.,2*t*np.pi/om),x)[-1].x[-1])
  # Poincare map
  pmap = lambda x : psi(1.,x)
  start = time.clock()
  # find periodic orbit
  z0 = rx.fp(pmap,zz,modes=[1,2])
  trjs0 = flow(hop, p, h, eps, (t0,tf), z0)
  hop.plot(trjs0,clf=False)
  
  print '      z0 = %s' % z0
  # linearize Poincare map
  J = rx.jac(pmap,z0)
  # compute eigenvalues of P-map
  print ' spec J  = %s' % ( np.linalg.eigvals(J) )
  print '|spec J| = %s' % np.abs( np.linalg.eigvals(J) )
  print np.array([(dd,np.abs(np.linalg.eigvals(rx.jac(pmap,z0,d=dd))).max()) for dd in np.logspace(-2,-6,40)])

  print "%0.2f sec for fp & jac " % (time.clock() - start)

  
  z = trjs0[-1].x[-1]
  sig = np.mod(ph0 + om*(tf-t0),2*np.pi) # == phf
  if len(z) == 2:
    y,dy = np.array(z)
    z = np.array([sig,y,dy])
  elif len(z) == 4:
    sig,y,dy,x,dx = np.array(z)
    z = np.array([sig,y,dy,x,dx])
  else:
    dbg()

  return hop,z,p

def do_anim():
  q=[1.,2.,10.,5.,2.,2.0,2*np.pi,np.pi,2.]
  m,M,k,b,l0,a,om,ph,g = q
  debug = False

  hop = Hop()
  p = hop.P(q,debug)

  ph0 = np.pi/2.
  t0 = ph0*np.pi/om
  tf = (10*2 + ph0)*np.pi/om
  n  = np.infty
  t = (t0,tf)
  j = 1
  z = np.array([0.2,1.75,0.,0.]) # nice transient
  #j = 2
  #z = np.array([1.0,0.0])
  #z = np.array([1.6,0.6])
  #z = np.array([1.4,0.6])
  p.j = j

  h = 1e-2
  eps = 1e-12
  Zeno = False

  import time; 
  
  start = time.clock()
  trjs = flow(hop, p, h, eps, (t0*om,tf*om), z)

  #hop.plot(trjs[-10:])
  hop.plot(trjs,fn='fig/hoptrj.eps')
  xlim = plt.xlim(); ylim = plt.ylim()
  plt.xlim(xlim); plt.ylim(ylim)

  q0 = q[:]; p0 = hop.P(q0,debug); p0.j = 1; z0 = np.array([0.25,1.75,0.,0.])
  hop.sch(3*np.pi/2,[0.25,2.15,0,0],p0,text=True,fn='fig/hopA.eps')
  #hop.sch(3*np.pi/2,[0.25,2.15,0,0],p0,text=False,fn='fig/hopAplain.eps')
  p0.j = 2
  hop.sch(np.pi/4,[1.5,0],p0,text=True,fn='fig/hopG.eps')
  #hop.sch(np.pi/4,[1.5,0],p0,text=False,fn='fig/hopGplain.eps')
  #hop.sch(t0,z,p,text=True,fn='fig/hop.pdf')
  p0.j = 1

  print "%0.2f sec for sim & plot" % (time.clock() - start)

  ph0 = 0#np.pi
  t0 = ph0*np.pi/om
  tf = (4*2 + ph0)*np.pi/om
  n  = np.infty
  t = (t0,tf)
  j = 2
  z = np.array([1.952,1.908])
  p.j = j

  h = 1e-2
  eps = 1e-12
  Zeno = False

  import time; 
  
  start = time.clock()
  trjs = flow(hop, p, h, eps, (t0*om,tf*om), z)

  hop.plot(trjs)
  xlim = plt.xlim(); ylim = plt.ylim()
  plt.xlim(xlim); plt.ylim(ylim)
  lt = plt.plot([t0,t0],ylim,'y',lw=5.,alpha=0.75)

  T,Z,O,P = rx.stack(trjs)

  fps = 20.
  fn  = 0

  pxh=400;
  import os;

  plt.ion()
  for t,z,p in zip(T,Z,P):
    for tt,zz in zip(t,z):
      if tt >= t0 + fn/fps:
        print '%4d : %0.2f , %0.2f' % (fn, t0+fn/fps, tt)
        #plt.ioff()
        plt.figure(1)
        lt[0].set_xdata([tt,tt])
        plt.draw()
        #plt.savefig('vid/trj%04d.png'%fn);
        plt.savefig('vid/trj%04d.png'%fn,dpi=pxh/4.);
        plt.figure(2)
        hop.sch(tt,zz,p)
        #plt.ion()
        plt.draw()
        plt.savefig('vid/hop%04d.png'%fn,bbox_inches='tight',pad_inches=-0.1,dpi=pxh/5.8)
        os.system('convert vid/hop%04d.png vid/trj%04d.png +append vid/hoptrj%04d.png'%(fn,fn,fn))
        fn += 1
    #break      
  os.system('rm vid/hop.mp4 vid/hoptrj.mp4')
  os.system('ffmpeg -r %d -sameq -i vid/hop%%04d.png vid/hop.mp4' % fps)
  os.system('ffmpeg -r %d -sameq -i vid/hoptrj%%04d.png vid/hoptrj.mp4' % fps)
  
def do_pmap():
  q=[1.,2.,10.,5.,2.,2.0,2*np.pi,np.pi,2.]
  m,M,k,b,l0,a,om,ph,g = q
  debug = False

  hop = Hop()
  p = hop.P(q,debug)

  ph0 = np.pi/2.
  t0 = ph0*np.pi/om
  tf = (10*2 + ph0)*np.pi/om
  n  = np.infty
  t = (t0,tf)
  j = 1
  z = np.array([0.2,1.75,0.,0.]) # nice transient
  #j = 2
  #z = np.array([1.0,0.0])
  #z = np.array([1.6,0.6])
  #z = np.array([1.4,0.6])
  p.j = j

  h = 1e-2
  eps = 1e-12
  Zeno = False

  import time; 
  
  start = time.clock()
  trjs = flow(hop, p, h, eps, (t0*om,tf*om), z)

  #hop.plot(trjs[-10:])
  hop.plot(trjs,fn='fig/hoptrj.eps')
  xlim = plt.xlim(); ylim = plt.ylim()
  plt.xlim(xlim); plt.ylim(ylim)

  q0 = q[:]; p0 = hop.P(q0,debug); p0.j = 1; z0 = np.array([0.25,1.75,0.,0.])
  hop.sch(3*np.pi/2,[0.25,2.15,0,0],p0,text=True,fn='fig/hopA.eps')
  #hop.sch(3*np.pi/2,[0.25,2.15,0,0],p0,text=False,fn='fig/hopAplain.eps')
  p0.j = 2
  hop.sch(np.pi/4,[1.5,0],p0,text=True,fn='fig/hopG.eps')
  #hop.sch(np.pi/4,[1.5,0],p0,text=False,fn='fig/hopGplain.eps')
  #hop.sch(t0,z,p,text=True,fn='fig/hop.pdf')
  p0.j = 1

  print "%0.2f sec for sim & plot" % (time.clock() - start)

  qq = q[:]
  pp = hop.P(qq); pp.j = trjs[-1].p.j
  zz = trjs[-1].x[-1]
  # Deformation of Poincare section
  psi = (lambda t,x : flow(hop,pp,h,eps,(ph0*np.pi,(2*t+ph0)*np.pi),x)[-1].x[-1])
  # Poincare map
  pmap = lambda x : psi(1.,x)
  start = time.clock()
  # find periodic orbit
  z0 = rx.fp(pmap,zz,modes=[1,2])
  print '      z0 = %s' % z0
  # linearize Poincare map
  J = rx.jac(pmap,z0)
  # compute eigenvalues of P-map
  print ' spec J  = %s' % ( np.linalg.eigvals(J) )
  print '|spec J| = %s' % np.abs( np.linalg.eigvals(J) )
  print 'max |spec J| as a function of perturbation:'
  #print np.array([(dd,np.abs(np.linalg.eigvals(rx.jac(pmap,z0,d=dd))).max()) for dd in np.logspace(-2,-6,10)])

  print "%0.2f sec for fp & jac " % (time.clock() - start)

if __name__ == "__main__":

  import sys
  args = sys.argv

  if '--fp' in args:
    hop,z0,p0 = do_fp()

  if '--pmap' in args:
    do_pmap()

  if '--anim' in args:
    do_anim()

