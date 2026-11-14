# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Positive contact-composite grade GP with heteroscedastic sample noise.

The model estimates a latent arithmetic grade using a lognormal moment
correction. Observation noise is excluded from latent prediction variance.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize

class LogGradeGP:
    def fit(self,xy,arithmetic_grade,log_noise_variance):
        x=np.asarray(xy,float);grade=np.asarray(arithmetic_grade,float);noise=np.asarray(log_noise_variance,float)
        if x.shape!=(len(grade),2) or noise.shape!=grade.shape or len(x)<5:raise ValueError('At least five paired 2D grade observations required')
        if not np.isfinite(x).all() or not np.isfinite(grade).all() or not np.isfinite(noise).all() or (grade<=0).any() or (noise<0).any():raise ValueError('Finite coordinates, positive grades and nonnegative noise required')
        self.center=x.mean(0);self.x=(x-self.center)/100;self.noise=noise
        self.y=np.log(grade)+.5*noise
        def system(p):
            theta=p[2];rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            local=self.x@rot.T/np.exp(p[:2]);K=np.exp(p[3])*np.exp(-3*cdist(local,local))
            K.flat[::len(K)+1]+=noise+np.exp(p[4])+1e-8
            factor=cho_factor(K,lower=True);one=cho_solve(factor,np.ones(len(x)));mean=float(one@self.y/one.sum())
            residual=self.y-mean;alpha=cho_solve(factor,residual)
            nll=.5*residual@alpha+np.log(np.diag(factor[0])).sum()+.5*np.log(one.sum())
            return nll,factor,mean,alpha,rot
        bounds=[(np.log(.2),np.log(5)),(np.log(.2),np.log(5)),(-np.pi/2,np.pi/2),(np.log(.01),np.log(4)),(np.log(.001),np.log(1))]
        best=None
        for theta in [0.,np.pi/4,-np.pi/4]:
            fit=minimize(lambda p:system(p)[0],[0,0,theta,np.log(.5),np.log(.05)],bounds=bounds,method='L-BFGS-B',options={'maxiter':150})
            if best is None or fit.fun<best.fun:best=fit
        self.parameters=best.x;_,self.factor,self.mean,self.alpha,self.rotation=system(best.x)
        self.fit_info={'ranges_m':(100*np.exp(best.x[:2])).tolist(),'angle_degrees':float(np.degrees(best.x[2])),
            'latent_log_variance':float(np.exp(best.x[3])),'residual_log_noise':float(np.exp(best.x[4])),
            'optimizer_success':bool(best.success),'training_composites':len(x),'log_mean':self.mean}
        return self

    def predict(self,xy):
        q=(np.asarray(xy,float)-self.center)/100
        a=self.x@self.rotation.T/np.exp(self.parameters[:2]);b=q@self.rotation.T/np.exp(self.parameters[:2])
        variance=np.exp(self.parameters[3]);K=variance*np.exp(-3*cdist(b,a))
        mu=self.mean+K@self.alpha
        # Include uncertainty in the fitted constant mean, conditional on covariance.
        one=cho_solve(self.factor,np.ones(len(a)));sol=cho_solve(self.factor,K.T)
        v=np.maximum(0,variance-np.sum(K*sol.T,axis=1)+(1-K@one)**2/one.sum())
        return {'mean_grade':np.exp(mu+.5*v),'median_grade':np.exp(mu),'latent_log_mean':mu,'latent_log_variance':v}
