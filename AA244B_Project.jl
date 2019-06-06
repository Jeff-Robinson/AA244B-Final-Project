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

## CONSTANTS ##
eps0 = 8.8542e-12
kc = 1/(4*pi*eps0)
e = 1.6022e-19 # C
me = 9.109e-31 # kg
mp = 1.6726e-27 # kg

mutable struct particle_species
    name::String
    q::Float64 # -> charge in units of e
    m::Float64 # -> mass in units of me
    xs::Array{Float64, 1} # -> positions in units of dx
    vs::Array{Float64, 1} # -> velocities in units of dx/dt
    Es::Array{Float64, 1} # -> electric field magnitude in units of kc*e/dx^2
end

mutable struct grid_node
    X::Float64 # -> location in units of dx
    phi::Float64 # -> electric potential in units of kc*e/dx
    E::Float64 # -> electric field magnitude in units of kc*e/dx^2
end

function make_particles(N, names, qs, ms) # N - no. particles per species
    particle_list = Array{particle_species, 1}()
    for i in 1:length(names)
        particle = particle_species(
            names[i],
            qs[i],
            ms[i],
            zeros(N[i]), # positions
            zeros(N[i]) # velocities
            )
        push!(particle_list, particle)
    end
    return particle_list
end

function make_nodes(L_sys, N_nodes)
    dx = L_sys/(N_nodes-1)
    node_list = Array{grid_node, 1}()
    for i in 0:N_nodes
        node = grid_node(i, 0.0, 0.0)
        push!(node_list, node)
    return node_list, dx
end

function update_node_phi(particle_list, node_list)
    for node in node_list
        node.phi = 0.0
    end
    for particle_spec in particle_list
        for i = 1:length(particle_spec.xs)
            particle_x = particle_spec.xs[i]
            node_idx_lo = floor(Int64, particle_x)
            node_idx_hi =  ceil(Int64, particle_x)

            node_lo = node_list[node_idx_lo + 1]
            node_hi = node_list[node_idx_hi + 1]

            node_lo.phi += particle_spec.q/(node_idx_hi - particle_x)
            node_hi.phi += particle_spec.q/(particle_x - node_idx_lo)
        end
    end
end

function update_node_E(node_list, BC)
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

function update_particle_E(particle_list, node_list)
    for particle_spec in particle_list
        for i = 1:length(particle_spec.xs)
            particle_x = particle_spec.xs[i]
            node_idx_lo = floor(Int64, particle_x)
            node_idx_hi =  ceil(Int64, particle_x)

            node_lo = node_list[node_idx_lo + 1]
            node_hi = node_list[node_idx_hi + 1]

            particle_spec.Es[i] = (node_idx_hi - particle_x)*node_lo.E
                                + (particle_x - node_idx_lo)*node_hi.E
        end
    end
end

