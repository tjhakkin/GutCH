#
# Solves 3-phase Cahn-Hilliard system representing the dewetting of a thin cohesive 
# Pdgfra^High layer sandwiched between Pdgfra^Low and epithelial phases.
#
# Numerical solver adapted from the FEniCS demo:
# https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html
# 
# For debugging: If dolfin complains about pkg-config, make sure the dolfin 
# pkg-config folder comes first in system path:
# import os
# print(os.environ['PATH'])

import random, time, sys, os
from dolfin import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps

# Time stepping family, e.g. THETA=1 -> backward Euler, THETA=0.5 -> Crank-Nicolson
THETA   = 0.5        


class InitialConditions(UserExpression):
    """
    Template for vector-valued initial conditions for solution function u.
    Note that eval() is just a dummy to be overwritten.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def set_seed(seed):
        random.seed(seed)
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
        values[3] = 0.0
    def value_shape(self):
        return (4,)


class PeriodicBoundary(SubDomain):
    """
    Subdomain for periodic boundary condition.
    """
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        # Map right boundary to left boundary.
        y[0] = x[0] - 1.0
        y[1] = x[1]


class CahnHilliardEquation(NonlinearProblem):
    """
    Class for interfacing with the Newton solver.
    """
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)


def init_model(InitialCondition, free_energy, N, eps, dt, T, kappa, M):
    """
    Initializes the model: Create domain; define the problem; initialize the
    solver and set initial conditions.
    
    Parameters:
        Expression for phase initial conditions.

    Returns:
        NewtonSolver object, CahnHilliardEquation object, 
        current solution function (u) and previous solution function (up).
    """
    # Form compiler options
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -march=native'
    parameters["form_compiler"]["representation"] = "uflacs"

    # Create mesh and build function space
    lx = 1.0
    ly = lx * N[1] / N[0] 
    mesh = RectangleMesh.create([Point(0.0, 0.0), Point(lx, ly)], [N[0]-1, N[1]-1], 
                                CellType.Type.quadrilateral)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, MixedElement(P1, P1, P1, P1), 
                       constrained_domain=PeriodicBoundary())

    # Solution spaces, mixed & split.
    # NOTE: For non-linear problems the unknown function u must be defined as Function
    # instead of TrialFunction.
    # https://fenicsproject.discourse.group/t/adding-expressions-with-non-matching-form-arguments-vs-v-1/1867/2

    u = Function(ME)
    c1, c2, mu1, mu2 = split(u)
    up = Function(ME)
    c1p, c2p, mu1p, mu2p = split(up)

    # Test functions
    q1, q2, v1, v2 = TestFunctions(ME)

    u_init = InitialCondition(degree=1)
    u.interpolate(u_init)
    up.interpolate(u_init)

    # Get the chemical potential derivatives dg/dc
    Fc1, Fc2 = free_energy()
    # Substitute variables, convert the SymPy output to Dolfin.
    k1 = kappa[0]
    k2 = kappa[1]
    k3 = kappa[2]
    c1 = variable(c1)
    c2 = variable(c2)
    Fc1 = eval(str(Fc1))    
    Fc2 = eval(str(Fc2))

    # The first line declares that ``c`` is a variable that some function
    # can be differentiated with respect to. The next line is the function
    # :math:`f` defined in the problem statement, and the third line
    # performs the differentiation of ``f`` with respect to the variable
    # ``c``.
    # 
    # It is convenient to introduce an expression for :math:`\mu_{n+\theta}`::

    # mu_(n+theta)
    mu1_mid = (1.0-THETA)*mu1p + THETA*mu1
    mu2_mid = (1.0-THETA)*mu2p + THETA*mu2

    # Weak statement of the equations
    L0 = (c1 - c1p)*q1*dx + dt*M[0]/kappa[0]*dot(grad(mu1_mid), grad(q1))*dx  # phase 1
    # L0 = (c1 - c1p)*q1*dx + dt*M[0]*dot(grad(mu1_mid), grad(q1))*dx  # phase 1
    L1 = mu1*v1*dx - 12*Fc1/eps*v1*dx - 0.75*kappa[0]*eps*dot(grad(c1), grad(v1))*dx

    L0 += (c2 - c2p)*q2*dx + dt*M[1]/kappa[1]*dot(grad(mu2_mid), grad(q2))*dx  # phase 2
    # L0 += (c2 - c2p)*q2*dx + dt*M[1]*dot(grad(mu2_mid), grad(q2))*dx  # phase 2
    L1 += mu2*v2*dx - 12*Fc2/eps*v2*dx - 0.75*kappa[1]*eps*dot(grad(c2), grad(v2))*dx

    L = L0 + L1

    # Jacobian of L with respect to u.
    J = derivative(L, u)
    # Alternative (what's the practical difference?):
    # du = TrialFunction(ME)
    # J = derivative(L, u, du)

    # Create nonlinear problem and Newton solver
    problem = CahnHilliardEquation(J, L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "residual"
    solver.parameters["relative_tolerance"] = 1e-8

    # The string ``"lu"`` passed to the Newton solver indicated that an LU
    # solver should be used.  The setting of
    # ``parameters["convergence_criterion"] = "incremental"`` specifies that
    # the Newton solver should compute a norm of the solution increment to
    # check for convergence (the other possibility is to use ``"residual"``,
    # or to provide a user-defined check). The tolerance for convergence is
    # specified by ``parameters["relative_tolerance"] = 1e-6``.

    return solver, problem, u, up


def solution_to_array(u, N):
    """
    Converts the solution function 'u' into concentration arrays for phases 1 
    and 2 suitable for plotting.

    TODO: We're upside-down by default, hence the flips at the end. Fix the
    initial conditions.

    Returns: 
        Phase concentration arrays p1, p2.
    """
    p1 = u.split()[0].compute_vertex_values()   # phase 1 concentrations
    p1 = np.resize(p1, [N[1], N[0]])
    p2 = u.split()[1].compute_vertex_values()   # phase 2 concentrations
    p2 = np.resize(p2, [N[1], N[0]])
    p1 = np.flip(p1, axis=0)
    p2 = np.flip(p2, axis=0)

    return p1,p2


# Create figure object. To make live plotting possible it seems necessary to 
# have this created during the initial execution of this cell.
# fig = plt.figure()

def init_figure(ax, u, N, dt, T):
    """
    Initialize the phases figure: Define the color map, plot initial phases.
    Note that only phases 1 and 2 are explicitly plotted; the 3rd phase is
    defined by the absence of phases 1 and 2.

    Returns:
        AxesImage objects for phases 1 and 2.
    """
    # Colormaps for the two phases. 
    cm0 = plt.get_cmap('PiYG').reversed()   # dark green to dark redish
    cm1 = plt.get_cmap('PRGn')              # dark purple to to darker green
    # cm0 = plt.get_cmap('RdBu_r')
    # cm1 = plt.get_cmap('PRGn')
    # Black-brown-green:    
    # cm0 = plt.get_cmap('BrBG')
    # cm1 = plt.get_cmap('RdGy')    

    # More reddish: "#D81B60"
    # Lighter purple: "#E305E5"
    #89058C

    # Gut simulations:
    colors0 = "#02E002", "#9E039F" # light green, red/purple
    colors1 = "#9E039F", "#004D40"  # red/purple, dark green
    
    # Aggregate simulations:
    # colors0 = "#02E002", "#FFFFFF" # light green, white
    # colors1 = "#FFFFFF", "#004D40"  # white, dark green
    
    # Inverted green aggregate simulations:
    # colors0 = "#004D40", "#FFFFFF" # light green, white
    # colors1 = "#FFFFFF", "#02E002"  # white, dark green

    cm0 = cmaps.get_continuous_cmap(colors0)
    cm1 = cmaps.get_continuous_cmap(colors1)

    # Modify the second phase colormap to have increasing transparency; 
    # this will give us a smooth transition boundary between the phases.
    cl1 = list(map(cm1, range(0,256))) 
    for i in range(0,256):
        cl1[i] = (cl1[i][0], cl1[i][1], cl1[i][2], i/255.0)

    cm1 = cm1.from_list('cm1', cl1, N=256)

    # Plot the initial phases.
    p1,p2 = solution_to_array(u, N)
    plot_phase1 = ax.imshow(p1, cmap=cm0, interpolation='spline16') # 'none')
    plot_phase2 = ax.imshow(p2, cmap=cm1, interpolation='spline16') # 'none')
    ax.axis('off')
    ax.set_title('Iteration: 0 / %d.' %(int(T/dt)), fontsize=10)

    return plot_phase1, plot_phase2


def simulate(IC, free_energy, fig, ax, N, eps, dt, T, kappa, M, rng_seed=2, title="", output=""):
    """
    Sets up the simulation, executes the simulation loop and updates the phases
    figure.

    Parameters:
        IC: Function for evaluting the initial conditions.
        free_energy: Adhesion energy function.
        fig: Output figure handle.
        ax: Axis of the output figure.
        N: Domain dimensions in number of nodes.
        eps: Epsilon parameter (interface thickness).
        dt: Time step.
        T: Total simulation time.
        kappa: Kappa parameters
        M: Phase mobilities.
        rng_seed: Define RNG seed for noise.
        title: Additional title string printed below the one set in the simulation loop.
        output: Name of the folder for storing intermediate stages (optional).

    Returns:
        Array of norm(u(n+1)-u(n)) for solution u at all time steps.
    """

    if (kappa[0]*kappa[1] + kappa[0]*kappa[2] + kappa[1]*kappa[2] <= 0.0):
        print("Ill-posed system (k1*k2 + k1*k3 + k2*k3 <= 0), giving up!")
        return 0

    if (kappa[0] < 0.0 or kappa[1] < 0.0 or kappa[2] < 0):
        print("Total spreading.")
    else:
        print("Partial spreading.")

    # Create output folder for storing the result images if needed.
    if (output != "" and not os.path.exists(output)):
        os.makedirs(output)

    # Initialize solver; define the problem; set initial conditions.
    InitialConditions.eval = IC
    InitialConditions.set_seed(rng_seed)
    solver, problem, u, up = init_model(InitialConditions, free_energy, N, eps, dt, T, kappa, M)

    # Initialize the figure. Note the figure object 'fig' itself already exists in 
    # global space.
    plot_phase1, plot_phase2 = init_figure(ax, u, N, dt, T)
    fig.canvas.draw()
    fig.canvas.flush_events()
    if (output != ""):
        fname = output + ("/0.png")
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # NOTE: Expanding bounding box vertically by 0.1% to account for some 
        # rounding errors in cropping that occasionally leave one row of pixels out.
        fig.savefig(fname, bbox_inches=bbox.expanded(1.0, 1.001))
        # fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=200)

    # Print initial phase volumes, mainly just for debugging.
    p1,p2 = solution_to_array(u, N)
    V = [np.sum(p1), np.sum(p2), N[0]*N[1]-np.sum(p1)-np.sum(p2)] # phase volumes
    print("Initial phase volumes: %f, %f, %f"  %(V[0], V[1], V[2]))

    dnorm = np.zeros(int(T/dt))    # norm(u(n+1)-u(n))

    t_start = time.time()   # current absolute time

    for i in range(1, int(T/dt)+1):
        up.vector()[:] = u.vector()

        nit = solver.solve(problem, u.vector())
        
        dnorm[i-1] = np.linalg.norm(u.vector()[:] - up.vector()[:])
        print("Iteration %d / %d, solution delta norm %lf" %(i, int(T/dt), dnorm[i-1]))

        # Update figure        
        p1,p2 = solution_to_array(u, N)
        plot_phase1.set_array(p1)
        plot_phase2.set_array(p2)
        ax.set_title('Iteration: %d / %d. T=%g, dt=%g. %s' 
                     %(i, int(T/dt), T, dt, title), fontsize=10)
        fig.canvas.draw()

        # If output folder given (assuming here it exists!), write the step into
        # a png file.
        if (output != ""):
            fname = output + ("/%d.png" %i) 
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(fname, bbox_inches=bbox.expanded(1.0, 1.001)) # dpi=194)
            # fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=200)

        print()
        sys.stdout.flush()

    # Write solution deltas if output folder given
    if (output != ""):
        file = open(output + "/dnorm.txt", 'w')
        for d in dnorm:
            file.write('%f\n' %d)

        file.close()

    # Final phase volumes.
    p1,p2 = solution_to_array(u, N)
    V = [np.sum(p1), np.sum(p2), N[0]*N[1]-np.sum(p1)-np.sum(p2)] # phase volumes
    print("Final phase volumes: %f, %f, %f"  %(V[0], V[1], V[2]))

    print("\nElapsed time %f seconds" %(time.time() - t_start))

    return dnorm
