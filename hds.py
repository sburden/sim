
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

# (c) Shai Revzen, U Penn 2010
import integro

class Obs(object):
    """  
    Obs  struct-like class for collecting observations

    Acts like a dict, but keys are members of the object.

    >>> a = Obs(food='bar', goo=7)
    >>> b = a.copy()
    >>> b.hoo = [1,2,3]
    >>> print a
    Obs(['food', 'goo'])
    >>> print b
    Obs(['food', 'hoo', 'goo'])
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    def copy(self):
        return Obs(**self.__dict__.copy())
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def __repr__(self):
        return 'Obs('+str(self.__dict__.keys())+')'

    def resample(self,dt):
        o = self.copy()
        t = o.t
        t0 = np.array([tt[0] for tt in t])
        T = np.arange(t[0][0],t[-1][-1],dt) 
        o.t = [T]
        for k in self.keys():
            x = self.__dict__[k]
            if k == 't':
                continue
            X = np.zeros((T.size,x[0].shape[1]))
            for n in range(T.size):
                nn = (t0 <= T[n]).nonzero()[0][-1]
                nnn = (t[nn] <= T[n]).nonzero()[0][-1]
                X[n,:] = x[nn][nnn]
            o.__dict__[k] = [X]

        return o

class HDS(object):
    def __init__(self,dt=.1):
        """
        sim.HDS  hybrid system superclass

        Full hybrid systems must override:
            .dyn, .obs, .trans, .evts;
        though trivial implementations are provided.

        To support analyzing periodic orbits, must override:
            .omap;
        then .ofind and .ojac become available.
        """
        self.t = []
        self.x = []
        self.q = []
        self.e = []

        self.dt = dt 

        self.name = 'HDS'

    def __call__(self, t, tf, x, q, Ne, clean=True, dbg=False):
        """
        HDS  executes hybrid system
        """
        if clean:
          self.t = []; self.x = []; self.q = []; self.e = []

        self.t += [np.array([t])]
        self.x += [np.array([x])]
        self.q += [np.array(q)]

        err = 0
        while ((self.t[-1][-1] < tf)
               and (len(self.q)-1)/2 < Ne and self.q[-1][0] >= 0
               and not (err < 0)):

            if dbg:
              print '%4.2f / %4.2f sec, %2d / %2d evts' % (self.t[-1][-1],tf,(len(self.q)-1)/2,Ne)

            t0 = self.t[-1][-1]
            x0 = self.x[-1][-1,:]
            q0 = self.q[-1]

            if dbg:
              print '0  %4.2f\n   %s\n   %s'%(t0,x0,q0)

            e = -1
            self.e = []
            self.ev = self.evts(q0)

            o = integro.odeDP5( lambda t,x,p : self.dyn(t, x, q0) )
            o.event = lambda t,x,trj : self.evt(t, x, q0)

            t,x,err = o(x0,t0,tf,dt=self.dt)

            assert err >= 0, "Check integro for details, err = %d" % err

            if any(self.e):
                e = self.e.index(True)
                t0,t1 = o.trj.range
                if self.ev[e](t0,x[-1])*self.ev[e](t1,x[-1]) <= 0:
                  o.refine(self.ev[e])

            self.t += [t.copy()]
            self.x += [x.copy()]
            self.q += [q0.copy()]

            if dbg:
              print '1  %4.2f\n   %s'%(t[-1],x[-1])

            if e >= 0:

                t,x,q = self.trans(t[-1],x[-1,:],q0,e)

                self.t += [np.array([t.copy()])]
                self.x += [np.array([x.copy()])]
                self.q += [q.copy()]

                if dbg:
                  print 'e  %4.2f\n   %s\n   %s'%(t,x,q)

        return self.t, self.x, self.q


    def dyn(self, t, x, q):
        """
        .dyn  evaluates system dynamics

        dx = hs.dyn(t, x, q)
        """
        return 0.*x

    def obs(self):
        """
        .obs  observes trajectory

        o = hs.obs()
        """
        return Obs(t=self.t,x=self.x)

    def trans(self, t, x, q, e):
        """
        .trans  executes hybrid transition for detected event

        t,x,q = hs.trans(t, x, q, e)
        """
        return t,x,q

    def evts(self, q):
        """
        .evts  returns event functions for given hybrid domain

        Event functions have the signature lambda t,x : ...
        and must evaluate to a real number so that:
            < 0 : no event
            >= 0 : event

        ev = hs.evts(q)
        """
        return []

    def evt(self, t, x, q):
        """
        .evt  evaluates event functions, stores result, returns flag

        f = hs.evt(t, x, q)
        """
        if len(t) == 2:
            t = t[-1]
            x = x[-1]
        self.e  = [ev(t,x) > 0 for ev in self.ev]
        return any(self.e)

    def dist(self, x,y):
      """
      .dist  distance in state space
      """
      return np.max(np.abs(x-y))

    def omap(self, x, args=[]):
        """
        .omap  Poincar\'{e} (or orbit) map

        x = hs.omap(x)
        """
        return x

    def ofind(self, x0, args=[], eps=1e-6, modes=[1,2], Ni=4, N=100, v=1):
        """
        .ofind  Find periodic orbit near x0 to within tolerance eps

        The algorithm has two modes:
            1. iterate .omap; keep result if error decreases initially
            2. run op.fsolve on .omap; keep if result is non-nan

        x = hs.ofind(x0)
        """
        suc = False

        if 1 in modes:
            # Iterate orbit map several times, compute error
            x = reduce(lambda x, y : self.omap(x, args), [x0] + range(Ni))
            xx = self.omap(x, args)
            e = self.dist(xx,x)
            e0 = self.dist(x,x0)
            # If converging to fixed point
            if e < e0:
                # Iterate orbit map
                n = 0
                while n < N-Ni and e > eps:
                    n = n+1
                    x = xx
                    xx = self.omap(x, args)
                    e = self.dist(xx,x)
                x0 = xx
                suc = True

        if 2 in modes:
            x = x0
            # Try to find fixed point using op.fsolve
            f = lambda x : self.omap(x, args) - x
            xx = op.fsolve(f, x)
            # If op.fsolve failed
            if not np.isnan(xx).any() or self.dist(xx,x) > e:
                x0 = xx
                suc = True

        if not suc:
            x0 = np.nan*x0

        return x0


    def ojac(self, x0, args=[], delta=None):
        """
        .ojac  Numerically approximate Jacobian to .omap at xi0

        If delta == None, attempt to guess a good delta using .omap

        J = hs.ojac(x0)
        """
        if not delta:
            x = self.omap(x0, args)
            delta = 1e4 * self.dist(x,x0)
            if not delta > 0:
                delta = 1e-6
        J = []
        N = len(x0)
        for k in range(N):
            d = np.zeros(N)
            d[k] = delta
            J += [0.5*(self.omap(x0+d, args) - self.omap(x0-d, args))/delta]

        return np.array(J)

    def __repr__(self):
        if self.t:
            return (self.name+
                    '\n  t='+str(self.t[-1][-1])+
                    '\n  x='+str(self.x[-1][-1,:])+
                    '\n  q='+str(self.q[-1]))
        else:
            return self.name

class BB(HDS):
    def __init__(self):
        """
        sim.BB  Bouncing Ball hybrid system
        """
        HDS.__init__(self)
        
        self.name = 'BB'

    def __call__(self, t, tf, x, q, Ne, c=0.9, g=9.81, h=0.):
        """
        BB()  executes hybrid system
        """
        return HDS.__call__(self, t, tf, x, [0, c, g, h], Ne)

    def dyn(self, t, x, q):
        """
        .dyn  evaluates system dynamics
        """
        return [x[1],-q[2]]

    def obs(self):
        """
        .obs  observes trajectory
        """
        o = Obs(t=self.t,E=[])
        for t,x,q in zip(self.t,self.x,self.q):
            o.E += [np.hstack([q[2]*x[:,0:1]])]

        self.o = o
        return self.o

    def trans(self, t, x, q, e):
        return t, np.array([x[0],-q[1]*x[1]]), q

    def evts(self, q):
        """
        .evts  returns event functions for given hybrid domain
        """
        return [lambda t,x : -(x[0] - q[3])]

if __name__ == "__main__":
    
    bb = BB()
    c = 0.8  # coefficient of restitution
    g = 9.81 # gravitational constant
    h = 0.   # ground height
    t,x,q = bb(0,1.,[1,0],0,10,c=c,g=g,h=h)
    o = bb.obs()
    oo = o.resample(0.025)

    xlim = [t[0][0],t[-1][-1]]

    plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(np.hstack(t),np.vstack(x),'.-')
    plt.xlim(xlim)
    plt.ylabel('Trajectory')
    plt.legend(('Position', 'Velocity'),ncol=2,loc=4)
    plt.subplot(2,1,2)
    plt.plot(np.hstack(o.t),np.vstack(o.E),'.-')
    plt.plot(np.hstack(oo.t),np.vstack(oo.E),'.-')
    plt.xlim(xlim)
    plt.ylabel('Energy')
    plt.xlabel('Time')



