import numpy as np
from scipy.special import j0,j1

def rotate_matrix(alpha,beta,gamma):
    '''
    rotate matrix for vector in euclid space
    defined by three anlges rotating about different axis in degrees
    alpha rotate about x-aixs counter-clock wise in y-z plane
    beta rotate about y-axis counter-clock wise in x-z plane
    gamma rotate about z-axis counter-clock wise in x-y plane
    '''
    alpha = np.radians(alpha)
    beta  = np.radians(beta)
    gamma = np.radians(gamma)
    
    Rx = np.array([(1, 0, 0,),
                   (0, np.cos(alpha), -np.sin(alpha)),
                   (0, np.sin(alpha),  np.cos(alpha))])
    Ry = np.array([( np.cos(beta), 0, np.sin(beta)),
                   (0, 1, 0),
                   (-np.sin(beta), 0, np.cos(beta))])
    Rz = np.array([(np.cos(gamma), -np.sin(gamma), 0),
                   (np.sin(gamma),  np.cos(gamma), 0),
                   (0, 0 ,1)])
    return np.matmul(np.matmul(Rx,Ry),Rz)

def shift_phase(qvec,rvec):
    '''
    the displacement of particle in real space will lead to phase difference in fourier space
    rvec is displacement of real space vector
    qvec is displacement of fourier space vector
    '''
    return np.exp(1j*np.dot(qvec,rvec))

def cylinder_form_factor(qvec,
                         density,
                         radius,
                         height):
    '''
    calculate the fourier transform of cylinder object
    radius and height are of cylinder
    density represent the uniform density within cylinder
    thus electron density = density * volume
    '''
    qr = np.hypot(qvec[0],qvec[1])
    V  = 2*np.pi*radius**2*height
    
    
    F  = 2*j0(qvec[2]*height/2)*j1(qr*radius)/qr/radius + 1j*0
    F *= density*V
    return F
    
def box_form_factor(qvec,density,a,b,c):
        '''
        Fourier transform of a box with edge length a, b, c.
        '''
        V = a*b*c
        F = np.sinc(qvec[0]*a)*np.sinc(qvec[1]*b)*np.sinc(qvec[2]*c)
        return F*density*V

def spherical_form_factor(qvec,density,r):
    q = np.linalg.norm(qvec)
    V = 4./3*np.pi*r**3
    return ((3*V*(np.sin(q*r)-q*r*np.cos(q*r))/((q*r)**3))**2)/V

def Atomic_form_factor(q,
                        a1,a2,a3,a4,
                        b1,b2,b3,b4,
                        c):
    f = a1*np.exp(-b1*(q/4/np.pi)**2) + \
        a2*np.exp(-b2*(q/4/np.pi)**2) + \
        a3*np.exp(-b3*(q/4/np.pi)**2) + \
        a4*np.exp(-b4*(q/4/np.pi)**2) + c
    return f

def vec_cal(a,b,c,alpha,beta,gamma):
    '''
    calculate the real space vector, and reciprocal space vector 
    here for convenience of fiber diffraction calculation, we assign c vector
    (fiber axis) same to z-axis
    '''
    a1 = a*np.sin(beta)
    a2 = 0.
    a3 = a*np.cos(beta)
    b3 = b*np.cos(alpha)
    b1 = (a*b*np.cos(gamma)-a3*b3)/a1
    b2 = np.sqrt(b**2-b1**2-b3**2)
    
    a_vec = np.array([a1, 0,a3])
    b_vec = np.array([b1,b2,b3])
    c_vec = np.array([ 0, 0, c])
    rvec = {'a' : a_vec,
            'b' : b_vec,
            'c' : c_vec}
    # s = 2*np.pi/V
    s = 2*np.pi/(c*(a1*b2-a2*b1))
    qa_vec = s*np.cross(b_vec,c_vec)
    qb_vec = s*np.cross(c_vec,a_vec)
    qc_vec = s*np.cross(a_vec,b_vec)
    qvec = {'A' : qa_vec,
            'B' : qb_vec,
            'C' : qc_vec}
    return rvec,qvec

def q_hkl(h,k,l,qvec):
    '''
    calculate the qvec of miller index
    '''
    qa_vec = qvec['A']
    qb_vec = qvec['B']
    qc_vec = qvec['C']
    q_vec = h*qa_vec+k*qb_vec+l*qc_vec
    q_hkl = np.linalg.norm(q_vec)
    return q_vec,q_hkl

def cellulose_form_factor(obj, q, rvec, obj_pos, abs_pos = False):
    '''
    calulcate scattering of one unit cell
    q,r include three base vector of unit cell in real space and reciprocal space
    obj_pos is object shift position to origin of unit cell
    abs_pos indicate the obj position is relative ratio of unit cell base vector or absolute real space position
    '''
    if obj == 'O':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 
                                                0.867, 32.9089, 0.2508])
    if obj == 'C':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([2.31, 20.8439, 1.02, 10.2075, 1.5886, 0.5687, 
                                                0.865, 51.6512, 0.2156])
    if obj == 'H':
        (a1,b1,a2,b2,a3,b3,a4,b4,c) = np.array([0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519, 
                                                0.049879, 2.20159, 0.001305])
    f = Atomic_form_factor(np.linalg.norm(q),a1,a2,a3,a4,b1,b2,b3,b4,c)
    
    if abs_pos == True:
        phase = np.exp(1j*np.dot(q,obj_pos))
    elif abs_pos == False:
        r = obj_pos[0]*rvec['a']+obj_pos[1]*rvec['b']+obj_pos[2]*rvec['c']
        phase = np.exp(1j*np.dot(q,r))
    return f*phase

#def cellulose_unit_cell_form_factor( q, rvec, 
#                               elements, elements_pos_x,
#                               elements_pos_y, elements_pos_z,abs_pos=False):
#    '''
#    calculate the sum of scattering from all objects within one unit cell
#    pars is dict include unit cell information and all the atoms information, such 
#    as elements, position within unit cell. These could be extracted from cif or pdb file.
#    here I specified save stlye for these in a dict within a json file.
#    q is vector, q = np.array([qx,qy,qz])
#    '''
#    
#    f = 0+0*1j
#    for _ in range(len(elements)):
#        if abs_pos == False:
#            f += form_factor(elements[_], q, rvec,
#                      np.array([elements_pos_x[_],elements_pos_y[_],elements_pos_z[_]]))
#        else:
#            f += form_factor(elements[_], q, rvec,
#                             np.array([elements_pos_x[_],elements_pos_y[_],elements_pos_z[_]]),abs_pos=True)
#    return f

def cellulose_hkl_coeff(rvec,qvec,h,k,l,pars):
    '''
    for Ibeta cif sym_pos is [[1,1,1][-1,-1,1+.5]]
    for Ialpha cif sym_pos if [[1,1,1]]
    '''
    fhkl = 0+0*1j
    qhkl_vec,qhkl = q_hkl(h,k,l,qvec)
    for _ in range(len(pars['atom_symbol'])):
        fhkl += cellulose_form_factor(pars['atom_symbol'][_],qhkl_vec,rvec,
                            np.array([pars['fract_x'][_],pars['fract_y'][_],pars['fract_z'][_]]))
        if pars['name'] == 'Ibeta':
            fhkl += cellulose_form_factor(pars['atom_symbol'][_],qhkl_vec,rvec,
                                 np.array([pars['fract_x'][_]*-1,pars['fract_y'][_]*-1,pars['fract_z'][_]+0.5]))
    return fhkl,qhkl

def peak_shape(q, p, mu, sigma):
    L = 2/np.pi/sigma/(1+(q-mu)**2/sigma**2)
    G = (4*np.log(2))**.5/sigma/np.pi**.5*np.exp(-4*np.log(2)*(q-i)**2/sigma**2)
    P = p*L+(1-p)*G
    return P/np.max(P)

def gaussian(q,mu,sigma):
    return np.exp(-(q-mu)**2/2/sigma**2)

def lorentz(q,mu,sigma):
    return 1/(1+(q-mu)**2/sigma**2)

def cellulose_structure_factor(q,h,k,l,pars,sigma):
    '''
    here calculate the powder diffraction pattern of cellulose
    which is spherical averaged intensity, should include lorentz correction with q**2
    '''
    
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    hv,kv,lv = np.meshgrid(np.arange(-h,h+1),np.arange(-k,k+1),np.arange(-l,l+1))
    hkl = np.vstack((hv.flatten(),kv.flatten(),lv.flatten()))
    S = np.zeros((len(q)))
    for _ in range(hkl.shape[1]):
        #print(hkl[0,_],hkl[1,_],hkl[2,_])
        fhkl,qhkl = cellulose_hkl_coeff(rvec,qvec,hkl[0,_],hkl[1,_],hkl[2,_],pars)
        if qhkl > np.max(q):
            pass
        else:
            S += np.abs(fhkl)**2*gaussian(q,qhkl,sigma)
    return S/q**2

def cellulose_layer_line(qr,h,k,l,pars,sigma):
    '''
    '''
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    # only calculate the h, k index for l layer
    hv,kv,lv = np.meshgrid(np.arange(-h,h+1),np.arange(-k,k+1),np.arange(l,l+1))
    hkl = np.vstack((hv.flatten(),kv.flatten(),lv.flatten()))
    S = np.zeros((len(qr)))
    for _ in range(hkl.shape[1]):
        #print(hkl[0,_],hkl[1,_],hkl[2,_])
        fhkl,qhkl = cellulose_hkl_coeff(rvec,qvec,hkl[0,_],hkl[1,_],hkl[2,_],pars)
        Qr_vec = hkl[0,_]*qvec['A']+hkl[1,_]*qvec['B']+hkl[2,_]*qvec['C']
        Qr = np.sqrt(Qr_vec[0]**2+Qr_vec[1]**2)
        if Qr > np.max(qr):
            pass
        else:
            S += np.abs(fhkl)**2*gaussian(qr,Qr,sigma)
    return S/qr


#############
#calculate ribbon like form factor
def hkl_coeff(rvec,qvec,h,k,l,pars,func_type='box'):
    '''
    '''
    fhkl = 0+0*1j
    qhkl_vec,qhkl = q_hkl(h,k,l,qvec)
    rot_mat = rotate_matrix(pars['rot_a'],
                            pars['rot_b'],
                            pars['rot_g'])
    #rot_mat = np.linalg.pinv(rot_mat)
    qhkl_vec = np.matmul(rot_mat.T,qhkl_vec)
    for _ in range(len(pars['particle'])):
        
        r = pars['particle_pos'][_][0]*rvec['a']+\
            pars['particle_pos'][_][1]*rvec['b']+\
            pars['particle_pos'][_][2]*rvec['c']
        phase = np.exp(1j*np.dot(r,qhkl_vec))
        if func_type == 'box':
            fhkl += box_form_factor(qhkl_vec,pars['particle_density'][_],
                                pars['la'],pars['lb'],pars['lc'])*phase
        elif func_type == 'sphere':
            fhkl += spherical_form_factor(qhkl_vec,pars['particle_density'][_],
                                pars['r'])*phase
    return fhkl,qhkl

#def sph_ave_form_factor(obj,pars,qr):
#    if obj = 'box':

def structure_factor(qr,h,k,l,pars,sigma,func_type='box'):
    '''
    '''
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    hv,kv,lv = np.meshgrid(np.arange(-h,h+1),np.arange(-k,k+1),np.arange(-l,l+1))
    hkl = np.vstack((hv.flatten(),kv.flatten(),lv.flatten()))
    S = np.zeros((len(qr)))
    for _ in range(hkl.shape[1]):
        fhkl,qhkl = hkl_coeff(rvec,qvec,hkl[0,_],hkl[1,_],hkl[2,_],pars,func_type=func_type) 
        if qhkl > np.max(qr):
            pass
        elif qhkl < 0.001:
            pass
        else:
            S += np.abs(fhkl)**2*gaussian(qr,qhkl,sigma)
    return S/qr**2

def layer_line_factor(qr,h,k,l,pars,sigma):
    '''
    '''
    a = pars['a']
    b = pars['b']
    c = pars['c']
    alpha = np.radians(pars['alpha'])
    beta  = np.radians(pars['beta'])
    gamma = np.radians(pars['gamma'])
    
    rvec,qvec = vec_cal(a,b,c,alpha,beta,gamma)
    # only calculate the h, k index for l layer
    hv,kv,lv = np.meshgrid(np.arange(-h,h+1),np.arange(-k,k+1),np.arange(l,l+1))
    hkl = np.vstack((hv.flatten(),kv.flatten(),lv.flatten()))
    S = np.zeros((len(qr)))
    for _ in range(hkl.shape[1]):
        fhkl,qhkl = hkl_coeff(rvec,qvec,hkl[0,_],hkl[1,_],hkl[2,_],pars)
        Qr_vec = hkl[0,_]*qvec['A']+hkl[1,_]*qvec['B']+hkl[2,_]*qvec['C']
        Qr = np.sqrt(Qr_vec[0]**2+Qr_vec[1]**2)
        if Qr > np.max(qr):
            pass
        else:
            S += np.abs(fhkl)**2*gaussian(qr,Qr,sigma)
    return S/qr
 
def dw_factor(q,a,sigma):
    return np.exp(-sigma**2*a**2*q**2)
        
def spherical_ave_form_factor(q,density,r):
    V = 4./3*np.pi*r**3
    return ((3*V*(np.sin(q*r)-q*r*np.cos(q*r))/((q*r)**3))**2)/V

def intensity(x,qr,h,k,l,func_type):
    pars = {}
    bkgd_p = x[0:6]
    dw_a = x[6]
    dw_sigma = x[7]
    scale1 = x[8]
    scale2 = x[9]
    sigma  = x[10]
    pars['a'] = x[11]
    pars['b'] = x[12]
    pars['c'] = x[13]
    pars['alpha'] = x[14]
    pars['beta']  = x[15]
    pars['gamma'] = x[16]
    pars['particle_pos'] = [x[17:20],x[20:23]]
    pars['particle'] = ['particle1','particle2']
    pars['particle_density'] = [10,10]
    pars['r'] = 10
    pars['rot_a'] = 0.
    pars['rot_b'] = 0.
    pars['rot_g'] = 0.
    S = structure_factor(qr,h,k,l,pars,sigma,func_type=func_type)
    F = spherical_ave_form_factor(qr,10,pars['r'])
    bkgd = np.polyval(bkgd_p,qr)
    I = ((1-dw_factor(q,dw_a,dw_sigma))*F+S*dw_factor(q,dw_a,dw_sigma)*scale1)*scale2+bkgd
    return I