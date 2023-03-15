# Modeling

# Activate the environment; Julia comes with a build in package manager, which can be called from the
# interactive REPL by `]` as a first command. We will activate and instantiate this environment ( meaning load all packages)
# Defined by the Manifest.toml.

using Pkg
Pkg.activate(dirname(@__FILE__))
Pkg.instantiate()
# Look at the output
Pkg.status()

# Load the necessary Packages

using ModelingToolkit

# Our goal is to model a simple Lotka Volterra system here
# We start by defining the necessary variables and parameters
# You can use unicode by typing in `\delta` and tab complete 

# The variables with annotation
# @ is julia syntax for a macro, which takes in all arguments and compiles the underlying expression. 
# This is really useful for domain specific languages
@variables t [description ="Time"] x(t)=1.0 [description="Prey species"] y(t)=0.1 [description ="Predator species"]
# Note that I do not store the output (a 3 element Vector of type Num). 
# We add Metadata (descripton) and intial values (1.0 for both species)
# Variables and Parameters created with the macro are stored in the scope, so now I can simply type
x 
# to get the variable.


# Now we define the parameters with the start values
@parameters α=1.0 β=1.0 γ=1.0 δ=1.0


# Now we define the time derivatives operator
D = Differential(t)

# Now we model the system of ordinary differential equations
eqs = [D(x) ~ α*x - β*x*y,
    D(y) ~ γ*x*y - δ*y,]

# And generate a named system (which helps to keep track of the variables)
@named sys = ODESystem(eqs)

# For a more beautiful output, we will use latexify
using Latexify
raw_tex_system = latexify(sys) 
render(raw_tex_system)

# Simulate the system

# We need to convert the system to an ODEProblem, which generates the code on the fly
prob = ODEProblem(sys)
# Next, we simulate the system using OrdinaryDiffEq.jl
using OrdinaryDiffEq

# We only add the simulation time span
solution = solve(prob, Tsit5(), tspan = (0.0, 10.0), saveat = 0.5)

# And plot the solution
using Plots

plot(solution)
# Maybe add a scatter plot here?
scatter!(solution, label = nothing)
# The `bang` operator ! indicates that the plot overlays the previous one (mutates it inplace)

# Identifiability
using StructuralIdentifiability

# Say we can only measure y
# define the output functions (quantities that can be measured)
@variables obs(t)

# We have one observation
measured_quantities = [obs ~ y,]
# Local identifiability with 0.99 probability
local_id_all = assess_local_identifiability(sys, measured_quantities = measured_quantities,
                                            p = 0.99)

# Gamma is not identifiable
print(local_id_all)

# Globally maybe?
global_id = assess_identifiability(sys, measured_quantities = measured_quantities)

# Gamma is not identifiable
print(global_id)


# Lets fit the model working with our simulated solution 
# We can simply remake the problem within our prediction
# Get the data
X = Array(solution) # Observations stored in a matrix where each row is a variable and each column a time point
t = solution.t # All time points
function objective(p)
    sol = solve(
        prob, Tsit5(), p = p, saveat = t, tspan = extrema(t)
    )
    X̂ = Array(sol)
    # Sum of Squares loss
    sum(abs2, X̂ .- X)
end

# Check the intial loss for random uniform parameters
objective(rand(4))

# Fit the objective
using Optim

res = Optim.optimize(objective, ones(4) .- 0.3*rand(4), LBFGS(), Optim.Options(show_trace = true))

# Look at the minimizer
minimum(res)
res.minimizer

# Optimize using Particle Swarm
res = Optim.optimize(objective, ones(4) .- 0.3*rand(4), ParticleSwarm(upper = ones(4), lower = zeros(4)), Optim.Options(show_trace = true))

# Look at the minimizer
minimum(res)
res.minimizer

# Bayesian

using Turing
using Distributions
using LinearAlgebra
# Instantiate the turing model

# We add noise to the observations
X_noisy = X .+ 0.1*randn(size(X))

plot(solution)
scatter!(t, X_noisy')

# This is basically taken from the turing tutorial
@model function fitlv(data, prob, t)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0.1, upper=2)
    γ ~ truncated(Normal(3.0, 0.5); lower=0.1, upper=4)
    δ ~ truncated(Normal(1.0, 0.5); lower=0.1, upper=2)

    # Simulate Lotka-Volterra model using the current parameters
    p = [α, β, γ, δ]
    predicted = solve(prob, Tsit5(); p=p, saveat=t, tspan = extrema(t))

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model = fitlv(X_noisy, prob, t)

# 1000 Samples, 3 Chains using NUTS sampler
chain = sample(model, NUTS(), MCMCSerial(), 1000, 3; progress=true)

# We can visualize the results of the three chains via StatsPlots, which 
# provides plotting recipes for chains
using StatsPlots

plot(chain)