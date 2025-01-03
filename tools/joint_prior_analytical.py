import numpy as np
import scipy

def Joint_prob_Xeff_Xp(Xeff,Xp,q = 1,amax = 1,flag_debug = False):
    r'''
    Returns Joint prior on Effective Spin parameter Xeff and Precessing Spin Parameter Xp,
    given Mass ratio q, Max allowed spin magnitude amax, $p(Xeff,Xp|q,amax)$.
    Assumes $q>0$ and $0<amax<=1.$ $q>1$ inputs are interepreted as $1/q$.
    if flag_debug is True, then the function shows some integration for debugging and returns 4 integrals separately.
    '''
    assert (amax <= 1)*(amax > 0),"Invalid amax. 0<amax<=1 is accepted."

    def S(z):
        r'''
        Spence's function or dilogarithm Li_2(z).
        '''
        return scipy.special.spence(1-z)

    def g(x0,a0,b0):
        r'''
        Primitive function of log(x-bi)/(x-a-i) (x>=0).
        '''
        ans = np.zeros(len(a0))+0j
        case_special = (b0 == 0)*(x0 == 0)
        case_b_less_1 = (np.abs(b0) < 1)*(np.logical_not(case_special))
        case_b_1_a_less_0 = (b0 == 1)*(a0 <= 0)
        case_otherwise = np.logical_not(case_b_less_1|case_b_1_a_less_0|case_special)

        for j, case in enumerate([case_special,case_b_less_1,case_b_1_a_less_0,case_otherwise]):
            if np.any(case):
                x = x0[case]
                a = a0[case]
                b = b0[case]
                if j == 0:
                    ans[case] = 0
                if j == 1:
                    ans[case] = np.log(x-b*1j)*np.log((a-x+1j)/(a+1j-b*1j))+S((x-b*1j)/(a+1j-b*1j))
                if j == 2:
                    ans[case] = 1/2*(np.log(x-a-1j))**2+S(-a/(x-a-1j))
                if j == 3:
                    ans[case] = np.log(a+1j-b*1j)*np.log(a-x+1j)-S((a-x+1j)/(a+1j-b*1j))
        return ans



    def G(x0,a0,b0):
        r'''
        \int_0^x((x^2+b^2))/(1+(x-a)^2) dx.
        '''

        case_positive_x = (x0 > 0)
        case_negative_x = (x0 < 0)
        ans = np.zeros(len(x0))

        for j,case in enumerate([case_positive_x,case_negative_x]):
            if np.any(case):
                x = x0[case]
                a = a0[case]
                b = b0[case]
                if j == 0:
                    if flag_debug:
                        print("Given Values to H(x,a,b):")
                        print('x',x)
                        print('a',a)
                        print('b',b)
                        print('')
                    ans[case] = np.imag(g(x,a,b)+g(x,a,-b))-np.imag(g(np.zeros(len(x)),a,b)+g(np.zeros(len(x)),a,-b))
                if j == 1:
                    if flag_debug:
                        print('Avoid x<0 by transposing.')
                    ans[case] = -G(-x,-a,b)
        return ans

    def F(_x,_a,_b,_c,_d):
        r'''
        \int_0^x b*log((x^2+c^2)/d^2)/(b^2+(x-a)^2) dx.
        '''
        if np.any(_d == 0):
            print('Something is wrong!')
            raise ValueError

        x0 = np.atleast_1d(_x)
        a0 = np.atleast_1d(_a)
        b0 = np.atleast_1d(_b)
        c0 = np.atleast_1d(_c)
        d0 = np.atleast_1d(_d)
        max_len = len(b0)
        arr = np.zeros((5,max_len))
        for j, X in enumerate([x0,a0,b0,c0,d0]):
            if X.shape[0] == 1:
                arr[j,:] = X[0]
            else:
                arr[j,:] = X[:]
        x0,a0,b0,c0,d0 = arr

        if flag_debug:
            print(x0,a0,b0,c0,d0)
        ans = np.zeros(len(b0))
        case = (b0 != 0)
        x = x0[case]
        a = a0[case]
        b = b0[case]
        c = c0[case]
        d = d0[case]
        ans[case] = G(x/b,a/b,c/b) + np.log(b*b/d/d)*(np.arctan(a/b)+np.arctan((x-a)/b))
        return ans


    def Joint_prob_Xeff_Xp_1(_Xeff,_Xp,_q,flag_debug):
        r'''
        Returns Joint prior on Effective Spin parameter Xeff and Precessing Spin Parameter Xp, given Mass ratio q.
        Max allowed spin magnitude amax is assumed to be amax=1.
        '''
        Xp = np.atleast_1d(_Xp)
        Xeff = np.atleast_1d(_Xeff)
        q = np.atleast_1d(_q)
        assert np.min(q) > 0
        q[q>1] = 1/q[q>1]

        zeta = (1+q)*Xeff
        nu = (4+3*q)/(3+4*q)

        max_len = np.max([Xp.shape[0],Xeff.shape[0],q.shape[0]])
        arr = np.zeros((5,max_len))
        for j, X in enumerate([Xeff,Xp,q,zeta,nu]):
            if X.shape[0] == 1:
                arr[j,:] = X[0]
            else:
                arr[j,:] = X[:]

        Xeff0,Xp0,q0,zeta0,nu0 = arr
        threshold = q0/nu0

        case_OoB = np.logical_or((Xp0 <= 0),(Xp0 >= 1))
        case_lsr_than_threshold = np.logical_and(np.logical_not(case_OoB),(Xp0 < threshold))
        case_gtr_than_threshold = np.logical_and(np.logical_not(case_OoB),(Xp0 >= threshold))

        #Joint PDF is a sum of 4 integrals
        int_1 = np.zeros(max_len)
        int_2 = np.zeros(max_len)
        int_3 = np.zeros(max_len)
        int_4 = np.zeros(max_len)

        for j,case in enumerate([case_OoB,
                                case_lsr_than_threshold,
                                case_gtr_than_threshold]):
            if np.any(case):
                Xeff = Xeff0[case]
                Xp = Xp0[case]
                q = q0[case]
                zeta = zeta0[case]
                nu = nu0[case]

                if j != 0:
                #Only int_2 have non-zero value for j = 2
                    max_2 = np.minimum(zeta+np.sqrt(1-Xp*Xp),q)
                    min_2 = np.maximum(zeta-np.sqrt(1-Xp*Xp),-q)
                    zero = (min_2 > max_2)
                    if np.any(zero):
                        max_2[zero] = min_2[zero]
                    if flag_debug:
                        print('Calculating 2nd integral for j = {}'.format(j))
                        print('x_max:',max_2)
                        print('x_min:',min_2)
                    int_2[case] = -(1+q)/8/q * (F(max_2,zeta,Xp,0,q)-F(min_2,zeta,Xp,0,q))

                #int_1,int_3,int_4 have none-zero value only if j = 1
                    if j == 1:

                        #int_1
                        max_1 = np.minimum(zeta+np.sqrt(1-Xp*Xp),np.sqrt(q*q-nu*nu*Xp*Xp))
                        min_1 = np.maximum(zeta-np.sqrt(1-Xp*Xp),-np.sqrt(q*q-nu*nu*Xp*Xp))
                        zero = (min_1 > max_1)
                        if np.any(zero):
                            max_1[zero] = min_1[zero]
                        if flag_debug:
                            print('Calculating 1st integral')
                            print('x_max:',max_1)
                            print('x_min:',min_1)
                        int_1[case] = (1+q)/8/q * (F(max_1,zeta,Xp,nu*Xp,q)-F(min_1,zeta,Xp,nu*Xp,q))

                        #int_3
                        max_3 = np.minimum(zeta+np.sqrt(q*q-nu*nu*Xp*Xp),np.sqrt(1-Xp*Xp))
                        min_3 = np.maximum(zeta-np.sqrt(q*q-nu*nu*Xp*Xp),-np.sqrt(1-Xp*Xp))
                        zero = (min_3 > max_3)
                        if np.any(zero):
                            max_3[zero] = min_3[zero]
                        if flag_debug:
                            print('Calculating 3rd integral')
                            print('x_max:',max_3)
                            print('x_min:',min_3)
                        int_3[case] = (1+q)*nu/8/q * (F(max_3,zeta,nu*Xp,Xp,1)-F(min_3,zeta,nu*Xp,Xp,1))

                        #int_4
                        max_4 = np.minimum(zeta+np.sqrt(q*q-nu*nu*Xp*Xp),q/q)
                        min_4 = np.maximum(zeta-np.sqrt(q*q-nu*nu*Xp*Xp),-q/q)
                        zero = (min_4 > max_4)
                        if np.any(zero):
                            max_4[zero] = min_4[zero]
                        if flag_debug:
                            print('Calculating 4th integral')
                            print('x_max:',max_4)
                            print('x_min:',min_4)
                        int_4[case] = -(1+q)*nu/8/q * (F(max_4,zeta,nu*Xp,0,1)-F(min_4,zeta,nu*Xp,0,1))

        if flag_debug:
            return int_1,int_2,int_3,int_4
        elif max_len == 1:
            return (int_1+int_2+int_3+int_4)[0]
        return int_1+int_2+int_3+int_4

    if flag_debug:
        int_1,int_2,int_3,int_4 = Joint_prob_Xeff_Xp_1(Xeff/amax,Xp/amax,q,flag_debug)
        return int_1/amax/amax,int_2/amax/amax,int_3/amax/amax,int_4/amax/amax
    return Joint_prob_Xeff_Xp_1(Xeff/amax,Xp/amax,q,flag_debug)/amax/amax
