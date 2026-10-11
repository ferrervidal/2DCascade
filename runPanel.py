from HS_panel import Airfoil
import airfoilCoords as afl
import matplotlib.pyplot as plt

# Create the airfoil for the profile of choice.
#coords=afl.DCA(20,15,300)
coords=afl.NACA4(2,4,12, 201)

# Create the airfoil for the profile of choice.
airfoil=Airfoil(coords)

# Solve.
swirl=0
airfoil.calculate(swirl)
airfoil.calculate_BL(1E5)


print("Cl: ", airfoil.Cl,"Cd: ", airfoil.Cd)
print("Cd visc: ", airfoil.Cd_visc, "Disp: ", airfoil.dispThick, "Mom.: ", airfoil.momThick)

plt.figure()
plt.plot(coords[0], coords[1], '-')
plt.plot(airfoil.stag[0], airfoil.stag[1], 'or')
plt.plot(airfoil.BL[0], airfoil.BL[1], '--r')
plt.plot()
plt.axis('equal')
plt.show()

plt.figure()
plt.plot(airfoil.xs, airfoil.Cp,'-')
plt.ylim([max(airfoil.Cp), min(airfoil.Cp)])
plt.xlabel("X/C")
plt.ylabel("Cp")
plt.show()



