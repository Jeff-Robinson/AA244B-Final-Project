#= 
AA244B Project - 1D PIC Code
Jeff Robinson - jbrobin@stanford.edu

Inputs for main solver:
L - length of 1D system
N_SP::Integer - number of species
DT - time step
NT - number of time steps to run
NG - number of grid points (power of 2)

Inputs for initialization:
N - number of particles


=#

using PyPlot
using Distributions

## CONSTANTS ##
eps0 = 8.854187812813*10^-12 # F/m
kc = 8987551787.3681764 # 1/(4*pi*eps0) # N-m^2/C^2
e = 1.602176634*10^-19 # C
kce = 1.439964516500946781275224*10^-9 # N-m^2/C
me = 9.109383701528*10^-31 # kg
mp = 1.6726219236951*10^-27 # kg
mpme = 1836.152673439956 # mp/me

mutable struct particle_species
    name::String
    q::Float64 # -> charge in units of e
    m::Float64 # -> mass in units of me
    xs::Array{Float64, 1} # -> positions in units of dx
    vs::Array{Float64, 1} # -> velocities in units of dx/dt
    vs_old::Array{Float64, 1} # -> old velocities in units of dx/dt
    Es::Array{Float64, 1} # -> electric field magnitude in units of kc*e/dx^2
    phis::Array{Float64, 1} # -> electric potential in units of kc*e/dx
end

mutable struct grid_node
    X::Float64 # -> location in units of dx
    phi::Float64 # -> electric potential in units of kc*e/dx
    E::Float64 # -> electric field magnitude in units of kc*e/dx^2
end

function make_particles(names, NPs, qs, ms) # NPs - no. particles per species
    particle_list = Array{particle_species, 1}()
    for i in 1:length(names)
        particle = particle_species(
            names[i],
            qs[i],
            ms[i],
            zeros(NPs[i]), # positions
            zeros(NPs[i]), # velocities
            zeros(NPs[i]), # old velocities
            zeros(NPs[i]), # Fields
            zeros(NPs[i]) # potentials
            )
        push!(particle_list, particle)
    end
    return particle_list
end

function make_nodes(L_sys, N_nodes)
    dx = L_sys/(N_nodes-1)
    node_list = Array{grid_node, 1}()
    for i in 0:N_nodes-1
        node = grid_node(i, # X
        0.0, # phi
        0.0 # E
        )
        push!(node_list, node)
    end
    return node_list, dx
end

function update_node_phi!(particle_list, node_list, BC)
    for node in node_list
        node.phi = 0.0
    end
    max_node_X = node_list[end].X
    N_nodes = length(node_list)
    for particle_spec in particle_list
        for i = 1:length(particle_spec.xs)
            particle_x = particle_spec.xs[i]
            if BC == "zero" && particle_x < 0 || particle_x > max_node_X
                continue
            elseif BC == "periodic"
                node_idx_lo = mod(floor(Int64, particle_x), N_nodes)
                node_idx_hi = mod( ceil(Int64, particle_x), N_nodes)
            else
                node_idx_lo = floor(Int64, particle_x)
                node_idx_hi =  ceil(Int64, particle_x)
            end

            node_lo = node_list[node_idx_lo + 1]
            node_hi = node_list[node_idx_hi + 1]

            node_lo.phi += particle_spec.q/(node_hi.X - particle_x)
            node_hi.phi += particle_spec.q/(particle_x - node_lo.X)
        end
    end
end

function update_node_E!(node_list, BC)
    N_nodes = length(node_list)
    for i = 1:N_nodes
        if i == 1 && BC == "zero"
            phi_lo = 0.0
        elseif i == 1 && BC == "periodic"
            phi_lo = node_list[end].phi
        else
            phi_lo = node_list[i-1].phi
        end

        if i == N_nodes && BC == "zero"
            phi_hi = 0.0
        elseif i == N_nodes && BC == "periodic"
            phi_hi = node_list[1].phi
        else
            phi_hi = node_list[i+1].phi
        end

        node_list[i].E = (phi_lo - phi_hi)/2
    end
end

function update_particle_Es!(particle_list, node_list, BC)
    max_node_X = node_list[end].X
    # min_node_X = node_list[1].X
    N_nodes = length(node_list)
    for particle_spec in particle_list
        for i = 1:length(particle_spec.xs)
            particle_x = particle_spec.xs[i]
            if BC == "zero" && particle_x < 0 || particle_x > max_node_X
                particle_spec.Es[i] = 0
                continue
            elseif BC == "periodic"
                node_idx_lo = mod(floor(Int64, particle_x), N_nodes)
                node_idx_hi = mod( ceil(Int64, particle_x), N_nodes)
            else
                node_idx_lo = floor(Int64, particle_x)
                node_idx_hi =  ceil(Int64, particle_x)
            end

            node_lo = node_list[node_idx_lo + 1]
            node_hi = node_list[node_idx_hi + 1]
            
            particle_spec.Es[i] = (node_hi.X - particle_x)*node_lo.E
                                + (particle_x - node_lo.X)*node_hi.E

            particle_spec.phis[i] = (node_hi.X - particle_x)*node_lo.phi
                                + (particle_x - node_lo.X)*node_hi.phi
        end
    end
end

function update_particle_vs!(particle_list, dt)
    for particle_spec in particle_list
        particle_spec.vs_old = particle_spec.vs
        particle_spec.vs .+= particle_spec.q/particle_spec.m*particle_spec.Es*dt
        # for i = 1:length(particle_spec.xs)
        #     particle_spec.vs[i] += particle_spec.q/particle_spec.m * particle_spec.Es[i] * dt
        # end
    end
end

function update_particle_xs!(particle_list, node_list, BC)
    N_nodes = length(node_list)
    for particle_spec in particle_list
        for i = 1:length(particle_spec.xs)
            if BC == "periodic"
                particle_spec.xs[i] = mod(particle_spec.xs[i] + particle_spec.vs[i], N_nodes)
            else
                particle_spec.xs[i] += particle_spec.vs[i]
            end
        end
    end
end

function init_particle_vs!(particle_list, node_list, BC, dt)
    update_node_phi!(particle_list, node_list, BC)
    update_node_E!(node_list, BC)
    update_particle_Es!(particle_list, node_list, BC)
    for particle_spec in particle_list
        for i = 1:length(particle_spec.xs)
            particle_spec.vs[i] += (-particle_spec.q/2)/particle_spec.m * particle_spec.Es[i] * dt
        end
    end
end

function init_cold_stationary(;
    names = ["electrons", "protons"], 
    NPs = [Int64(1e6), Int64(1e6)], 
    qs = [-1, 1], 
    ms = [1, mpme], 
    L_sys = 10^-6, # m
    N_nodes = 10^5, 
    dt = 5e-9, # s
    BC
    )
    particle_list = make_particles(names, NPs, qs, ms)
    node_list, dx = make_nodes(L_sys, N_nodes)
    N_species = length(names)
    for k = 1:N_species
        # N_particles = NPs[k]
        # particle_x_intrv = (L_sys/N_particles)/dx
        # particle_x_shift = k/(N_species+1) * particle_x_intrv
        # for i = 1:N_particles
            # particle_list[k].xs[i] = particle_x_intrv * i + particle_x_shift
        # end
        particle_list[k].xs = rand(Uniform(0, L_sys), NPs[k])/dx
    end
    init_particle_vs!(particle_list, node_list, BC, dt)
    return particle_list, node_list, dx, dt
end

function run_sim(BC = "periodic", N_steps_max = 100, N_steps_save = 25)
    particle_list, node_list, dx, dt = init_cold_stationary(BC = BC)
    N_steps = 0
    fig_idx = 0
    KE = []
    PE = []
    while N_steps < N_steps_max
        update_node_phi!(particle_list, node_list, BC)
        update_node_E!(node_list, BC)
        update_particle_Es!(particle_list, node_list, BC)
        update_particle_vs!(particle_list, dt)
        update_particle_xs!(particle_list, node_list, BC)
        N_steps += 1

        #= ENERGY CONSERVATION =#
        #= Potential Energy qΦ =#

        #= Kinetic Energy m*Vold*Vnew/2 =#

        #= PLOTTING =#
        if N_steps == 1 || mod(N_steps, N_steps_save) == 0
            fig_idx += 1
            fig1 = figure(fig_idx) # phase space
            (ax1, ax2) = fig1.subplots(nrows = 2, ncols = 1)
            ax1.set_xlabel("Particle Position, m")
            ax1.set_ylabel("Particle Velocity, m/s")
            ax2.set_xlabel("Particle Position, m")
            ax2.set_ylabel("Particle Velocity, m/s")
            # fig2 = figure(1) # velocity distribution
            # ax2 = fig2.subplots()
            fig3 = figure(fig_idx + 2) # grid potential, field
            (ax3, ax4) = fig3.subplots(nrows = 2, ncols = 1)
            ax3.set_xlabel("Node Position, m")
            ax3.set_ylabel("Node Potential, V")
            ax4.set_xlabel("Node Position, m")
            ax4.set_ylabel("Node Electric Field, V/m")
            # Plot phase space representation of particles
            ax1.scatter(particle_list[1].xs * dx, particle_list[1].vs * dx/dt)
            ax2.scatter(particle_list[2].xs * dx, particle_list[2].vs * dx/dt)
            # Plot grid potential & field
            node_Xs = [node.X * dx for node in node_list]
            node_phis = [node.phi * kce/dx for node in node_list]
            node_Es = [node.E * kce/dx^2 for node in node_list]
            ax3.scatter(node_Xs, node_phis)
            ax4.scatter(node_Xs, node_Es)
        end
    end
end