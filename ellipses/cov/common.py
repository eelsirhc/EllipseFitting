from numpy import zeros

def derivativeXcenter(a,b,c,d,e,delta):
    gradXcenter = zeros((6,1))
    gradXcenter[0] =  (4*c*(2*c*d-b*e))/delta**2
    gradXcenter[1] =  (b**2*e+4*a*c*e-4*b*c*d)/delta**2
    gradXcenter[2] =  (2*b*(b*d-2*a*e))/delta**2
    gradXcenter[3] =  (2*c)/delta
    gradXcenter[4] =   -b/delta
    gradXcenter[5] =   0
    return gradXcenter


def derivativeYcenter(a,b,c,d,e,delta):
    gradYcenter = zeros((6,1))
    gradYcenter[0] =  (2*b*(b*e-2*c*d))/delta**2
    gradYcenter[1] =  (b**2*d+4*a*c*d-4*a*b*e)/delta**2
    gradYcenter[2] =  (4*a*(2*a*e-b*d))/delta**2
    gradYcenter[3] =  -b / delta
    gradYcenter[4] =  2*a/delta
    gradYcenter[5] =  0
    return gradYcenter


def derivativeTau(a,b,c):
    # add condition for when a==c and b = 0
    gradTau = zeros((6,1))
    gradTau[0] = -b/(2*(b**2+(a-c)**2))
    gradTau[1] =  (a-c)/(2*(b**2+(a-c)**2))
    gradTau[2] =  b/(2*(b**2+(a-c)**2))
    gradTau[3] =  0
    gradTau[4] =  0
    gradTau[5] =  0
    return gradTau


def derivativeVplus(a,b,c,d,e,f,psi,lambdaPlus,delta):
    gradVplus = zeros((6,1))
    gradVplus[0] =  computeDerivativeVplusA(a,b,c,d,e,f,psi,lambdaPlus,delta)
    gradVplus[1] =  computeDerivativeVplusB(a,b,c,d,e,f,psi,lambdaPlus,delta)
    gradVplus[2] =  computeDerivativeVplusC(a,b,c,d,e,f,psi,lambdaPlus,delta)
    gradVplus[3] =  computeDerivativeVplusD(a,b,c,d,e,f,psi,lambdaPlus,delta)
    gradVplus[4] =  computeDerivativeVplusE(a,b,c,d,e,f,psi,lambdaPlus,delta)
    gradVplus[5] =  computeDerivativeVplusF(a,b,c,d,e,f,psi,lambdaPlus,delta)
    return gradVplus


def computeDerivativeVplusA(a,b,c,d,e,f,psi,lambdaPlus,delta):
    part1 = 1/(2*lambdaPlus*delta)
    part2 = (psi/(lambdaPlus*delta))**(-0.5)
    part3 = 4*c*f - e**2 + 4*delta**(-1)*c*psi
    part4 = (psi/(2*lambdaPlus))*(1 + ((c - a)/((a - c)**2 + b**2)**0.5))
    dVplusA = part1*part2*(part3 - part4)
    return dVplusA


def computeDerivativeVplusB(a,b,c,d,e,f,psi,lambdaPlus,delta):
    part1 = 1/(2*lambdaPlus*delta);
    part2 = (psi/(lambdaPlus*delta))**(-0.5);
    part3 = d*e - 2*b*f - 2*delta**(-1)*b*psi;
    part4 = (b*psi)/((2*lambdaPlus)*((a - c)**2 + b**2)**0.5);
    dVplusB =  part1*part2*(part3 + part4);                                                     
    return dVplusB


def computeDerivativeVplusC(a,b,c,d,e,f,psi,lambdaPlus,delta):
    part1 = 1/(2*lambdaPlus*delta)
    part2 = (psi/(lambdaPlus*delta))**(-0.5)
    part3 = 4*a*f - d**2 + 4*delta**(-1)*a*psi
    part4 = (psi/(2*lambdaPlus))*(1 + ((a - c)/((a - c)**2 + b**2)**0.5))
    dVplusC =part1*part2*(part3 - part4)
    return dVplusC


def computeDerivativeVplusD(a,b,c,d,e,f,psi,lambdaPlus,delta):
    part1 = (psi/(lambdaPlus*delta))**(0.5)
    part2 = (b*e - 2*c*d)/(2*psi)
    dVplusD = part1*part2
    return dVplusD


def computeDerivativeVplusE(a,b,c,d,e,f,psi,lambdaPlus,delta):                                                    
    part1 = (psi/(lambdaPlus*delta))**(0.5)
    part2 = (b*d - 2*a*e)/(2*psi);
    dVplusE = part1*part2;
    return dVplusE


def computeDerivativeVplusF(a,b,c,d,e,f,psi,lambdaPlus,delta):
    part1 = -(2*lambdaPlus)**-1;
    part2 = (psi/(lambdaPlus*delta))**(-0.5);
    dVplusF = part1*part2;
    return dVplusF


def derivativeVminus(a,b,c,d,e,f,psi,lambdaMinus,delta):
    gradVminus = zeros((6,1))
    gradVminus[0] = computeDerivativeVminusA(a,b,c,d,e,f,psi,lambdaMinus,delta)
    gradVminus[1] = computeDerivativeVminusB(a,b,c,d,e,f,psi,lambdaMinus,delta)
    gradVminus[2] = computeDerivativeVminusC(a,b,c,d,e,f,psi,lambdaMinus,delta)
    gradVminus[3] = computeDerivativeVminusD(a,b,c,d,e,f,psi,lambdaMinus,delta)
    gradVminus[4] = computeDerivativeVminusE(a,b,c,d,e,f,psi,lambdaMinus,delta)
    gradVminus[5] = computeDerivativeVminusF(a,b,c,d,e,f,psi,lambdaMinus,delta)
    return gradVminus


def computeDerivativeVminusA(a,b,c,d,e,f,psi,lambdaMinus,delta):
    part1 = 1/(2*lambdaMinus*delta)
    part2 = (psi/(lambdaMinus*delta))**(-0.5)
    part3 = 4* c*f - e**2 + 4*delta**(-1)*c*psi
    part4 = (psi/(2*lambdaMinus))*(1 - ((c - a)/((a - c)**2 + b**2)**0.5))
    dVminusA = part1*part2*(part3 - part4)
    return dVminusA


def computeDerivativeVminusB(a,b,c,d,e,f,psi,lambdaMinus,delta):
    part1 = 1/(2*lambdaMinus*delta)
    part2 = (psi/(lambdaMinus*delta))**(-0.5)
    part3 = d*e - 2*b*f - 2*delta**(-1)*b*psi
    part4 = (b*psi)/((2*lambdaMinus)*((a - c)**2 + b**2)**0.5)
    dVminusB = part1*part2*(part3 - part4)
    return dVminusB


def computeDerivativeVminusC(a,b,c,d,e,f,psi,lambdaMinus,delta):
    part1 = 1/(2*lambdaMinus*delta)
    part2 = (psi/(lambdaMinus*delta))**(-0.5)
    part3 = 4*a*f - d**2 + 4*delta**(-1)*a*psi
    part4 = (psi/(2*lambdaMinus))*(1 - ((a - c)/((a - c)**2 + b**2)**0.5))
    dVminusC =  part1*part2*(part3 - part4)
    return dVminusC


def computeDerivativeVminusD(a,b,c,d,e,f,psi,lambdaMinus,delta):
    part1 = (psi/(lambdaMinus*delta))**(0.5)
    part2 = (b*e - 2*c*d)/(2*psi)
    dVminusD =  part1*part2
    return dVminusD


def computeDerivativeVminusE(a,b,c,d,e,f,psi,lambdaMinus,delta):
    part1 = (psi/(lambdaMinus*delta))**(0.5)
    part2 = (b*d - 2*a*e)/(2*psi)
    dVminusE = part1*part2
    return dVminusE

def computeDerivativeVminusF(a,b,c,d,e,f,psi,lambdaMinus,delta):
    part1 = -(2*lambdaMinus)**-1
    part2 = (psi/(lambdaMinus*delta))**(-0.5)
    dVminusF = part1*part2
    return dVminusF
