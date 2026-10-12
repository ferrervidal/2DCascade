# 2D Cascade Analysis Code
Ferrer-Vidal, 2026

This module implements a Hess-Smith panel code to model 2D cascades of arbitraty solidity. Simple BL treatment is used to calculate loss and deviation. This code has not been validated for viscous flow. 

The airfoils module has utilities to create blading coordinates. This is still in development as the NACA utility is not currently set-up to model cascades. It can however be used to model isolated airfoils. 

The airfoil build function takes profile coordinates and solidity. Solidity is optional. Omitting it entails an isolated airfoil calculation. 

The airfoil calculate function calculates the potential flow solution for the desired swirl angle. For isolated airfoils, swirl is the AoA. The code internally uses velocity and chord values set to 1. 

The calculate_BL functions performs the viscous calculation, which is still WIP. The only input for viscous calculations is the chord-based Reynolds number. The code currently does not iterate the potential flow solution based on the state of the BL. 