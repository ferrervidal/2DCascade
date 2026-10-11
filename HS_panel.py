import numpy as np
"""
Hess-Smith Panel Code
"""

class Airfoil:

    def __init__(self,coords, solidity=None):

        X=coords[0]
        Y=coords[1]
        self.Cp=None
        self.Cl=None
        self.Cd=None
        self.solidity=solidity
        self.NoB=30

        self.panels=self.__createPanels__(X,Y)

        # Attributes below apply only for viscous solution.
        self.nu=None
        self.separated=False
        self.suctionSide=None
        self.pressureSide=None

    def __createPanels__(self, xcoord, ycoord):

        if self.solidity is not None:
            pitch=1/self.solidity
            NoB=self.NoB
        else:
            pitch=0
            NoB=0

        # Influence panels is a list of lists. Each inner list corresponds the same panel location in each blade.
        influence_panels=[]
        # Main panels are the only ones we will solve the source and vorticity for. 
        main_panels=[]
        for n in range(len(xcoord)):
            k=(n+1) % len(xcoord)
            thisPanelList=[]           
            for blade in range(-NoB,NoB+1):
                yshift=blade*pitch
                thisPanel=Panel(xcoord[n], xcoord[k], ycoord[n]+yshift, ycoord[k]+yshift)
                thisPanelList.append(thisPanel)
                if blade==0:
                    main_panels.append(thisPanel)
            influence_panels.append(thisPanelList)

        # Calculate the influence coefficients, which depend on all panels. 
        for thepanel in main_panels:
            thepanel.panelInfluence(influence_panels)

        return main_panels

    def calculate(self, alpha):
        # Solve for panel sources and single vorticity. 
        Vinf=1
        #Calculate the Vx and Vy components due to free-stream.
        alphar=alpha/180.0*np.pi
        Vx=Vinf*np.cos(alphar)
        Vy=Vinf*np.sin(alphar)

        # Calculate the component of free-stream velocity that acts normal to each panel.
        DEPS=np.zeros(len(self.panels)+1)
        COEFFS=np.zeros([len(self.panels)+1, len(self.panels)+1])
        for n, thepanel in enumerate(self.panels):
            DEPS[n]=-(-Vx*np.sin(thepanel.beta)+Vy*np.cos(thepanel.beta))
            # For each panel, get the influence coefficients. 
            COEFFS[n,:-1]=thepanel.coeffsNorm
            # Append the vortcity column:
            COEFFS[n,-1]=thepanel.vortNormCoeff
            #Assemble the Kutta row:
            COEFFS[-1,n]=self.panels[0].coeffsTang[n]+self.panels[-1].coeffsTang[n]

        # Last coeff in Kutta row:
        COEFFS[-1,-1]=self.panels[0].vortTangCoeff+self.panels[-1].vortTangCoeff
        # Kutta dependent    
        DEPS[-1]=-((Vx*(np.cos(self.panels[0].beta)+np.cos(self.panels[-1].beta)))+(Vy*(np.sin(self.panels[0].beta)+np.sin(self.panels[-1].beta))))


        # Solve the linear problem to enforce flow tangency. 
        sol=np.linalg.solve(COEFFS, DEPS)

        # For each panel:
        # - assign the value of vorticity from the previous solution. 
        # - assign the normal and tangential velocities. 
        # - assign the value of Cp.
        # - assign the valu of Cnc, the normal force coefficient times panel length
        for n,panel in enumerate(self.panels):
            panel.vort=sol[-1]
            panel.source=sol[n]
            panel.Vnorm=np.dot(panel.coeffsNorm, sol[:-1])+panel.vort*panel.vortNormCoeff+(-Vx*np.sin(panel.beta)+Vy*np.cos(panel.beta))
            panel.Vtang=np.dot(panel.coeffsTang, sol[:-1])+panel.vort*panel.vortTangCoeff+(Vx*np.cos(panel.beta)+Vy*np.sin(panel.beta))
            panel.Cp=1-(panel.Vtang/Vinf)**2
            panel.Cnc=-panel.Cp*panel.length
    

        # Calculate physical parameters for output. 
        cl=0
        cd=0
        cps=np.zeros(len(self.panels))
        xs=np.zeros(len(self.panels))
        ys=np.zeros(len(self.panels))
        for n,thepanel in enumerate(self.panels):
            cl+=thepanel.Cnc*np.cos(-alpha/180*np.pi+thepanel.beta)
            cd+=thepanel.Cnc*np.sin(-alpha/180*np.pi+thepanel.beta)
            cps[n]=thepanel.Cp
            xs[n]=thepanel.xcntr
            ys[n]=thepanel.ycntr

        
        self.Cp=cps
        self.Cl=cl
        self.Cd=cd
        self.xs=xs
        self.ys=ys

        normal_residual = np.array([p.Vnorm for p in self.panels])
        print("Maximum normal residual:", np.max(np.abs(normal_residual)))
        net_source = sum(p.source * p.length for p in self.panels)
        print("Net source strength:", net_source)

    def calculate_BL(self, Re):
        self.separated = False

        for p in self.panels:
            p.theta = 0.0
            p.thw = 0.0
            p.H = 0.0
            p.disp = 0.0
            p.Cfc = 0.0

        self.nu=1/Re 
        self.Re=Re
        if self.Cp is None:
            raise RuntimeError("Potential flow solution not calculated.")
        
        self.__calcStag__()
        self.__thwaites__(self.suctionSide)
        self.__thwaites__(self.pressureSide[::-1])
        

        self.Cd_visc=np.sum(p.Cdf for p in self.panels)
        x=list(p.xcntr_disp for p in self.panels)
        y=list(p.ycntr_disp for p in self.panels)
        self.BL=(x,y)
        self.dispThick=max(p.disp for p in self.panels)
        self.momThick=max(p.theta for p in self.panels)

    def __calcStag__(self):

        panels = self.panels
        Vtang = np.array([p.Vtang for p in panels])

        # Find sign changes between adjacent panels.
        sign_changes = []

        for n in range(len(panels) - 1):
            if Vtang[n] * Vtang[n + 1] < 0:
                sign_changes.append(n)

        if len(sign_changes) == 0:
            raise ValueError(
                "No stagnation-point sign change found in tangential velocity."
            )

        if len(sign_changes) > 1:
            raise ValueError(
                "More than one tangential-velocity sign change was found."
            )

        # The sign change occurs between n and n+1.
        n = sign_changes[0]

        p1 = panels[n]
        p2 = panels[n + 1]

        fraction = -p1.Vtang/(p2.Vtang - p1.Vtang)

        xstag = p1.xcntr + fraction*(p2.xcntr - p1.xcntr)
        ystag = p1.ycntr + fraction*(p2.ycntr - p1.ycntr)

        self.stag = (xstag, ystag)

        self.pressureSide = panels[:n + 1]
        self.suctionSide = panels[n + 1:]
    
    def __thwaites__(self, panels):

        cumulative_integral = 0.0
        total_s=0.0

        # Use positive edge-speed magnitudes
        U_previous = abs(panels[0].Vtang)

        for n in range(1, len(panels)):

            panel = panels[n]
            U_current = abs(panel.Vtang)

            if n==1:
                # For first panel after stagnation, distance to stagnation point calculated.
                ds=((panels[1].xcntr-self.stag[0])**2+(panels[1].ycntr-self.stag[1])**2)**0.5
            else:
                # Distance between adjacent panel centers
                ds = 0.5*(panels[n - 1].length + panel.length)
            # Keep track of total distance
            total_s+=ds

            # Trapezoidal integration of integral(Ue^5 ds)
            cumulative_integral += (0.5*(U_previous**5 + U_current**5)*ds)

            if U_current == 0.0:
                panel.theta = 0.0
            else:
                panel.theta = np.sqrt(0.45*self.nu*cumulative_integral/U_current**6)
            # Thwaites' parameter lambda.
            thw=panel.theta**2/self.nu*(U_current-U_previous)/ds
            panel.thw=thw
            if thw<-0.09: # Flow has separated.
                self.separated=True
                print("WARNING: Laminar flow (Thwaites') has separated at:", panel.xcntr, panel.ycntr)
                break
            panel.Sthw=(thw+0.09)**0.62
            # Shape factor
            panel.H=2+4.14*(0.25-thw)-83.5*(0.25-thw)**2+854*(0.25-thw)**3-3337*(0.25-thw)**4+4576*(0.25-thw)**5
            # Displacement thickness
            panel.disp=panel.H*panel.theta

            if self.__michel__(abs(panel.Vtang),total_s,panel.theta):
                print("Flow transition at: ", panel.xcntr, panel.ycntr)
                break

            # Panel contribution to overall viscous drag. Drag in global x-dir.
            panel.Cdf=((self.nu*panel.Sthw/(panel.disp*0.5)))* panel.Vtang**2* np.sign(panel.Vtang)* np.cos(panel.beta)* panel.length
            

            # Find displaced panel center (no relaxation, for info only)
            dispVec=panel.disp*panel.norm
            panel.xcntr_disp=panel.xcntr+dispVec[0]
            panel.ycntr_disp=panel.ycntr+dispVec[1]

            U_previous = U_current

    def __michel__(self,V,s,theta):
        # Michel transition criterion
        Re_s=V*s/self.nu
        Re_theta=V*theta/self.nu
        Re_tr=1.1714*(1+22400/Re_s)*Re_s**0.46
        if Re_theta>= Re_tr:
            return True
        else:
            return False



class Panel:

    # Currently assuming panel center is both vortex location and collocation.
    def __init__(self, x1, x2, y1,y2):
        delx=x2-x1
        dely=y2-y1
        self.length=np.sqrt(delx**2+dely**2)
        if self.length == 0:
            raise ValueError("Panel has zero length")
        self.xcntr=(x1+x2)/2
        self.ycntr=(y1+y2)/2
        self.start=(x1, y1)
        self.end=(x2, y2)
        # Calculate the angle of the panel normal. 
        # atan2 gives the correct sign.
        self.beta = np.arctan2(dely, delx)
        # Panel normal vector
        self.norm=np.array([-dely/self.length, delx/self.length])

        self.vort=None
        self.source=None
        self.Vnorm=None
        self.Vtang=None
        self.TE=False

        #Viscous parameters
        self.theta = 0.0
        # Other Thwaites' parameters at first panels.
        self.thw=0.0
        self.Sthw=0.0
        self.H=0.0
        self.disp=0.0
        self.Cdf=0.0
        self.xcntr_disp=self.xcntr
        self.ycntr_disp=self.ycntr
        

    def panelInfluence(self, influence_panels):
        # Calculates the influence coefficients for the sources and vortex.

        self.coeffsNorm=np.zeros(len(influence_panels))
        self.coeffsTang=np.zeros(len(influence_panels))
        self.vortNormCoeff=0
        self.vortTangCoeff=0

        for n in range(len(influence_panels)):
            coeffNorm=0
            coeffTang=0
            # Each list within influence_panels is the same panel, but each in a different blade.

            # Get some trig functions for speed, as they are the same for all panels:
            
            source_panels = influence_panels[n]
            panelbeta=source_panels[0].beta
            cosbeta=np.cos(panelbeta)
            sinbeta=np.sin(panelbeta)
            xstart=source_panels[0].start[0] # This is the same for the same panel across all blades, since only y changes.
            length=source_panels[0].length
            dbeta = panelbeta - self.beta
            ystart = np.array([p.start[1] for p in source_panels])


            xrel = (self.xcntr - xstart)*cosbeta + (self.ycntr - ystart)*sinbeta
            yrel = -(self.xcntr - xstart)*sinbeta + (self.ycntr - ystart)*cosbeta

            # Identify which image panel is the target panel itself
            self_mask = np.array([p is self for p in source_panels])

            # Allocate coefficients for all image panels.
            norm_coeffs = np.zeros(len(source_panels))
            tang_coeffs = np.zeros(len(source_panels))

            # Calculate the nonsingular coefficients for all panels.
            normal_source = (1/(2*np.pi)*(np.arctan2(yrel, xrel - length)- np.arctan2(yrel, xrel)))
            tangential_source = (1/(4*np.pi)* np.log((yrel**2 + xrel**2)/ ((xrel - length)**2 + yrel**2)))

            # Convert coefficients to absolute coordinate system.
            norm_coeffs = (tangential_source*np.sin(dbeta)+ normal_source*np.cos(dbeta))
            tang_coeffs = (tangential_source*np.cos(dbeta)- normal_source*np.sin(dbeta))

            # Replace the singular self-influence with the analytical limiting values
            norm_coeffs[self_mask] = 0.5
            tang_coeffs[self_mask] = 0.0

            # Sum the contributions from all periodic images
            coeffNorm = np.sum(norm_coeffs)
            coeffTang = np.sum(tang_coeffs)

            # We pass these coeffs to a list of coefficients for later use in the solver matrix.
            self.coeffsTang[n]=coeffTang
            self.coeffsNorm[n]=coeffNorm
            # For the vortex strength, we add the influences, as vortex strengths are the same.
            # Luckily, the previously calculated coeffs can be used for vorticity as below.
            self.vortNormCoeff+=coeffTang
            self.vortTangCoeff+=-coeffNorm


            # # Previous inner loop instead of arrays. This is slower.
            # for panel in influence_panels[n]:
            #     # Calculate the location of this panel relative to the location of every other panel.
            #     xrel=(self.xcntr-xstart)*cosbeta+(self.ycntr-panel.start[1])*sinbeta
            #     yrel=-(self.xcntr-xstart)*sinbeta+(self.ycntr-panel.start[1])*cosbeta

            #     if panel is self:
            #         # Self-influence if the panel is itself.
            #         coeffNorm+=0.5
            #         coeffTang+=0
            #     else:
            #         # Source flow normal and tangential velocity influence coefficient relative to each panel.
            #         coeffNormSource=1/(2*np.pi)*(np.arctan2(yrel, xrel-length)-np.arctan2(yrel, xrel))
            #         coeffTangSource=1/(4*np.pi)*np.log((yrel**2+xrel**2)/((xrel-length)**2+yrel**2))
            #         # Convert these coefficients to the absolute coordinate system.
            #         coeffNorm+=coeffTangSource*np.sin(panelbeta-self.beta)+coeffNormSource*np.cos(panelbeta-self.beta)
            #         coeffTang+=coeffTangSource*np.cos(panelbeta-self.beta)-coeffNormSource*np.sin(panelbeta-self.beta)
            




