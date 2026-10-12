from panel import Cascade
import airfoils as afl

""" INPUT: Cascade parameters"""
solidity=1.2 # Solidity (if none specied, single airfoil assumed)
swirl_in=50 # [deg] If single airfoil, this is equivalent to AoA.
swirl_out=30 # [deg]
Re_c=1E7 # Chord-based Reynolds. Only needed for viscous calculation.
incidence=-5 # deg
max_thickness=8 #Percent of chord.
# Create the airfoil for the profile of choice.
coords=afl.quadCamber(swirl_in-incidence,swirl_out, max_thickness,numpts=300, thickness_model="NACA4")

"""SOLVE"""
# Create the cascade for the airfoil and solidity of choice.
csc=Cascade(coords, solidity)
# Calculate potential flow via Hess-Smith panel code.
csc.calculate_pot(swirl_in)
# Calculate viscous effects with Re_c of choice.
csc.calculate_BL(Re_c)
# Show results
csc.output()





