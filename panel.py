import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

"""
Hess-Smith Panel Code
"""

class Cascade:

    def __init__(self,coords, solidity=None):

        self.X=coords[0]
        self.Y=coords[1]
        self.Cp=None
        self.Cl=None
        self.Cd=None
        self.solidity=solidity
        self.NoB=30

        self.panels=self.__createPanels__(self.X,self.Y)

        # Attributes below apply only for viscous solution.
        self.nu=None
        self.separated=False
        self.suctionSide=None
        self.pressureSide=None
        self.lamSepPoints=[]
        self.turbSepPoints=[]
        self.transPoints=[]

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

    def calculate_pot(self, alpha):
        self.swirl_in=alpha
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
        # Calculate Vte=V2
        Vte = 0.5*(abs(self.panels[-1].Vtang) +abs(self.panels[0].Vtang))
        self.DH=Vte # Since V1=1.

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
            p.H1 = 0.0
            p.Cf = 0.0
            p.disp = 0.0
            p.Cdf = 0.0
            p.state = "laminar"
            p.xcntr_disp = p.xcntr
            p.ycntr_disp = p.ycntr

        self.nu=1/Re 
        self.Re=Re
        if self.Cp is None:
            raise RuntimeError("Potential flow solution not calculated.")
        
        self.__calcStag__()
        self.__BLmarch__(self.suctionSide)
        self.__BLmarch__(self.pressureSide[::-1])
        self.__wakeloss__()
        self.__mixed_flow_angle__()
        
        # Calculate diffusion factor (per original definition)
        Vmax_ss = max(abs(panel.Vtang) for panel in self.suctionSide)
        # Note DH=V2 since V1=1.
        self.DF= 1.0 - self.DH/Vmax_ss

        self.Cd_visc=sum(p.Cdf for p in self.panels)
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

        for p in panels[n + 1:]: p.label="Suction Side" 
        for p in panels[:n + 1:]: p.label="Pressure Side"

        self.pressureSide = panels[:n + 1]
        self.suctionSide = panels[n + 1:]
    
    def __BLmarch__(self, panels):

        cumulative_integral = 0.0
        total_s=0.0
        turbulent = False

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
            if not turbulent:
                # Thwaites' parameter lambda.
                thw=panel.theta**2/self.nu*(U_current-U_previous)/ds
                panel.thw=thw
                if thw < -0.09: # Flow has separated.
                    self.separated=True
                    panel.state="separated"
                    print("WARNING: Laminar flow (Thwaites') has separated at:",panel.label, " : ", panel.xcntr, panel.ycntr)
                    self.lamSepPoints.append((panel.xcntr, panel.ycntr))
                    break

                panel.Sthw=(thw+0.09)**0.62
                # Shape factor
                panel.H=2+4.14*(0.25-thw)-83.5*(0.25-thw)**2+854*(0.25-thw)**3-3337*(0.25-thw)**4+4576*(0.25-thw)**5
                panel.disp=panel.H*panel.theta
                panel.Cf=self.nu*panel.Sthw/(0.5*panel.disp)
                panel.Cdf=self.__friction_drag__(panel, panel.Cf)
                panel.state="laminar"

                # Find displaced panel center (no relaxation, for info only)
                dispVec=panel.disp*panel.norm
                panel.xcntr_disp=panel.xcntr+dispVec[0]
                panel.ycntr_disp=panel.ycntr+dispVec[1]

                if self.__michel__(U_current,total_s,panel.theta):
                    print("Flow transition at: ", panel.label, " : ", panel.xcntr, panel.ycntr)
                    self.transPoints.append((panel.xcntr, panel.ycntr))
                    turbulent = True
                    panel.state="turbulent"
                    panel.H=1.4
                    panel.H1=self.__H1_from_H__(panel.H)
                    panel.disp=panel.H*panel.theta
                    panel.Cf=self.__head_Cf__(U_current,panel.theta,panel.H)
                    panel.Cdf=self.__friction_drag__(panel, panel.Cf)

                    dispVec=panel.disp*panel.norm
                    panel.xcntr_disp=panel.xcntr+dispVec[0]
                    panel.ycntr_disp=panel.ycntr+dispVec[1]

            else:
                self.__head_step__(panel, panels[n-1], U_previous,
                                   U_current, ds)

                if panel.state == "separated":
                    break

            U_previous = U_current

    def __head_step__(self, panel, previous_panel, U_previous,
                      U_current, ds):
        """Advance one turbulent Head-method panel."""

        if previous_panel.theta <= 0.0:
            raise RuntimeError("Head method requires positive momentum thickness.")

        dUds=(U_current-U_previous)/ds

        theta_previous=previous_panel.theta
        H1_previous=previous_panel.H1

        dtheta_1, dH1_1 = self.__head_rhs__(
            theta_previous, H1_previous, U_previous, dUds
        )

        theta_predict=max(theta_previous+ds*dtheta_1, 1.0e-14)
        H1_predict=H1_previous+ds*dH1_1
        
        Hsep=3.3 #turbulent separation threshold.
        H1sep=self.__H1_from_H__(Hsep)
        if H1_predict <= H1sep:
            self.separated=True
            panel.state="separated"
            print("WARNING: Turbulent flow (Head) has separated at: ", panel.label, " : ", panel.xcntr, panel.ycntr)
            self.turbSepPoints.append((panel.xcntr, panel.ycntr))
            return

        dtheta_2, dH1_2 = self.__head_rhs__(
            theta_predict, H1_predict, U_current, dUds
        )

        panel.theta=max(
            theta_previous+0.5*ds*(dtheta_1+dtheta_2),
            1.0e-14
        )
        panel.H1=H1_previous+0.5*ds*(dH1_1+dH1_2)

        if panel.H1 <= H1sep:
            self.separated=True
            panel.state="separated"
            print("WARNING: Turbulent flow (Head) has separated at: ", panel.label, " : ", panel.xcntr, panel.ycntr)
            self.turbSepPoints.append((panel.xcntr, panel.ycntr))
            return

        panel.H=self.__H_from_H1__(panel.H1)
        panel.Cf=self.__head_Cf__(U_current,panel.theta,panel.H)
        panel.disp=panel.H*panel.theta
        panel.Cdf=self.__friction_drag__(panel, panel.Cf)
        panel.state="turbulent"

        # Find displaced panel center (no relaxation, for info only)
        dispVec=panel.disp*panel.norm
        panel.xcntr_disp=panel.xcntr+dispVec[0]
        panel.ycntr_disp=panel.ycntr+dispVec[1]

    def __head_rhs__(self, theta, H1, Ue, dUds):
        """Return dtheta/ds and dH1/ds for Head's method."""

        if theta <= 0.0 or Ue <= 0.0 or H1 <= 3.300001:
            raise RuntimeError("Invalid state in Head boundary-layer method.")

        H=self.__H_from_H1__(H1)
        Cf=self.__head_Cf__(Ue,theta,H)

        dtheta_ds=0.5*Cf-(H+2.0)*theta/Ue*dUds

        CE=0.0306*(H1-3.0)**(-0.6169)
        dH1_ds=CE/theta-H1*(dtheta_ds/theta+dUds/Ue)

        return dtheta_ds, dH1_ds

    def __head_Cf__(self, Ue, theta, H):
        Re_theta=Ue*theta/self.nu
        return 0.246*10.0**(-0.678*H)*Re_theta**(-0.268)

    def __H1_from_H__(self, H):
        if H <= 1.6:
            return 3.3+0.8234*(H-1.1)**(-1.287)
        return 3.3+1.5501*(H-0.6778)**(-3.064)

    def __H_from_H1__(self, H1):
        if H1 <= 3.300001:
            raise RuntimeError("Head entrainment variable reached separation limit.")

        H1_switch=self.__H1_from_H__(1.6)

        if H1 >= H1_switch:
            return 1.1+(0.8234/(H1-3.3))**(1.0/1.287)

        return 0.6778+(1.5501/(H1-3.3))**(1.0/3.064)

    def __friction_drag__(self, panel, Cf):
        # Positive contribution to drag in the global x-direction.
        return Cf*panel.Vtang**2*np.sign(panel.Vtang)*np.cos(panel.beta)*panel.length

    def __wakeloss__(self):
        """
        Estimate the mixed-out total-pressure loss coefficient
        from the trailing-edge boundary-layer momentum thicknesses.
        """

        # SS & PS TE momentum thickness
        theta_ss = self.panels[-1].theta
        theta_ps = self.panels[0].theta

        # Edge velocity at each trailing-edge panel.
        # abs() avoids sign differences caused by panel orientation.
        Ue_ss = abs(self.panels[-1].Vtang)
        Ue_ps = abs(self.panels[0].Vtang)

        # Momentum-thickness contribution normalized by V^2, which is one.
        theta_wake = (Ue_ss**2 * theta_ss+ Ue_ps**2 * theta_ps)

        # Fully mixed wake loss estimate.
        self.loss = 2.0 * theta_wake / (1/self.solidity)

    def __mixed_flow_angle__(self):
        """
        Estimate the mixed-out flow angle using trailing-edge
        displacement thicknesses.
        """

        # Panel tangent unit vectors
        t_ss = np.array([np.cos(self.panels[-1].beta),np.sin(self.panels[-1].beta)])
        t_ps = np.array([np.cos(self.panels[0].beta),np.sin(self.panels[0].beta)])

        # Global velocity vectors at the two trailing-edge panels.
        # The signed Vtang accounts for panel orientation.
        V_ss = self.panels[-1].Vtang*t_ss
        V_ps = self.panels[0].Vtang*t_ps

        # Common Kutta velocity, approximated by averaging both TE vectors.
        V_kutta_vector = 0.5*(V_ss + V_ps)
        V_kutta = np.linalg.norm(V_kutta_vector)

        if V_kutta <= 0.0:
            raise ValueError("Trailing-edge velocity is zero")

        e_kutta = V_kutta_vector/V_kutta

        # Flow directions associated with each displaced boundary layer
        e_ss = V_ss/np.linalg.norm(V_ss)
        e_ps = V_ps/np.linalg.norm(V_ps)

        # Effective blockage fractions
        B_ss = self.panels[-1].disp/(1/self.solidity)
        B_ps = self.panels[0].disp/(1/self.solidity)

        if B_ss < 0.0 or B_ps < 0.0:
            raise ValueError("Displacement thickness must be non-negative")

        if B_ss + B_ps >= 1.0:
            raise ValueError("Trailing-edge blockage exceeds the blade pitch")

        # Pitch-averaged mixed-out velocity
        Vmix = V_kutta*((1.0 - B_ss - B_ps)*e_kutta+ B_ss*e_ss+ B_ps*e_ps)

        self.swirl_out = np.arctan2(Vmix[1], Vmix[0])/np.pi*180

    def __michel__(self,V,s,theta):
        # Michel transition criterion
        Re_s=V*s/self.nu
        Re_theta=V*theta/self.nu
        Re_tr=1.1714*(1+22400/Re_s)*Re_s**0.46
        if Re_theta>= Re_tr:
            return True
        else:
            return False

    def output(self):

        # Table data
        table_data = [
            ["Solidity", "Re_c", "Cl", "Cd_visc", "Loss Coef.", "Flow Angle In [deg]", "Flow Angle Out [deg]", "DH", "DF"],
            [f"{self.solidity:.2f}",f"{self.Re:.2E}",f"{self.Cl:.3f}", f"{self.Cd_visc:.4f}", f"{self.loss:.4f}", 
            f"{self.swirl_in:.2f}", f"{self.swirl_out:.2f}", f"{self.DH:.2f}", f"{self.DF:.2f}"]
            ]

        table_data=list(map(list, zip(*table_data)))
        # Create figure with GridSpec for custom layout
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 2, width_ratios=[1.2, 2])  # 2 rows, 2 columns

        # --- Left column: Table ---
        ax_table = fig.add_subplot(gs[:, 0])  # Span both rows
        ax_table.axis("off")  # Hide axes

        table = ax_table.table(
            cellText=table_data[1:],  # Data rows
            colLabels=table_data[0],  # Header row
            loc="center",
            cellLoc="center"
        )

        # Format table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # Highlight header row
        # for key, cell in table.get_celld().items():
        #     if key[0] == 0:  # Header row
        #         cell.set_facecolor("#cccccc")
        #         cell.set_text_props(weight="bold")

        # --- Right column, top plot ---
        ax_plot1 = fig.add_subplot(gs[0, 1])
        ax_plot1.set_title("Airfoil & BL")
        ax_plot1.plot(self.X, self.Y, '-')
        ax_plot1.plot(self.stag[0], self.stag[1], 'or', label="Stagnation point.")
        ax_plot1.plot(self.BL[0], self.BL[1], '--r', label="BL displacement hickness,")
        if self.lamSepPoints:
            x, y = zip(*self.lamSepPoints)
            ax_plot1.plot(x, y, 'ok', label="Laminar separation.")
        if self.turbSepPoints:
            x, y = zip(*self.turbSepPoints)
            ax_plot1.plot(x, y, 'ob', label="Turbulent separation")
        if self.transPoints:
            x, y = zip(*self.transPoints)
            ax_plot1.plot(x, y, 'oy', label="Transition point.")
        ax_plot1.plot()
        ax_plot1.legend()
        ax_plot1.axis('equal')
     
        # --- Right column, bottom plot ---
        ax_plot2 = fig.add_subplot(gs[1, 1])
        ax_plot2.set_title("Loading")
        ax_plot2.plot(self.xs, self.Cp,'-', label="Potential flow")
        #ax_plot2.ylim([max(self.Cp), min(self.Cp)])
        ax_plot2.set_xlabel("X/C")
        ax_plot2.set_ylabel("Cp")
        ax_plot2.invert_yaxis()
        ax_plot2.legend()

        # Adjust layout
        plt.tight_layout()
        plt.show()


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
        # Laminar and turbulent boundary-layer parameters.
        self.thw=0.0
        self.Sthw=0.0
        self.H=0.0
        self.H1=0.0
        self.Cf=0.0
        self.disp=0.0
        self.Cdf=0.0
        self.state="laminar"
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
            




