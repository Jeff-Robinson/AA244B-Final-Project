#= 
AA244B Project - 1D PIC Code
Jeff Robinson - jbrobin@stanford.edu
=#

using PyPlot
using Distributions
using LinearAlgebra
using Random
using DataFrames
using GLM
using FFTW

## CONSTANTS ##
kb = 1.3806485279*10^-23 #J/K
eps0 = 8.854187812813*10^-12 # F/m
# kc = 8987551787.3681764 # 1/(4*pi*eps0) # N-m^2/C^2
e = 1.602176634*10^-19 # C
# kce = 1.439964516500946781275224*10^-9 # N-m^2/C
me = 9.109383701528*10^-31 # kg
mp = 1.6726219236951*10^-27 # kg
mpme = 1836.152673439956 # mp/me

mutable struct PIC_particle_species
    name::String
    q::Float64 # -> charge in units of e
    m::Float64 # -> mass in units of me
    xs::Array{Float64, 1} # -> positions in units of dx
    vs::Array{Float64, 1} # -> velocities in units of dx/dt
    vs_old::Array{Float64, 1} # -> old velocities in units of dx/dt
    Es::Array{Float64, 1} # -> electric field magnitude in units of e/eps0*dx^2
    phis::Array{Float64, 1} # -> electric potential in units of e/eps0*dx
end

mutable struct PIC_node_list
    Xs::Array{Float64, 1} # -> locations in units of dx
    charges::Array{Float64, 1} # -> net charges in units of e
    phis::Array{Float64, 1} # -> electric potential in units of e/eps0*dx
    Es::Array{Float64, 1} # -> electric field magnitude in units of e/eps0*dx^2
end

function make_particles(names, NPs, qs, ms)
    particle_list = Array{PIC_particle_species, 1}()
    N_species = length(names)
    for i in 1:N_species
        particle = PIC_particle_species(
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
    return particle_list, N_species
end

function make_nodes(N_nodes)
    node_list = PIC_node_list(
        [x for x in 0:N_nodes-1], # Xs
        zeros(N_nodes), # charges
        zeros(N_nodes), # phis
        zeros(N_nodes) # Es
        )
    return node_list
end

#= BIRDSALL & LANGDON EQ 2-6 (1-2) =#
function update_node_charge!(particle_list, node_list, N_nodes, BC)
    node_list.charges = zeros(N_nodes)

    if BC == "zero"
        for particle_spec in particle_list
            for i = 1:length(particle_spec.xs)
                if particle_spec.xs[i] < 0 || particle_spec.xs[i] > node_list.Xs[end]
                    continue
                end
                node_idx_lo = floor(Int64, particle_spec.xs[i]) + 1

                node_list.charges[node_idx_lo] += particle_spec.q*(node_list.Xs[node_idx_lo + 1] - particle_spec.xs[i])
                node_list.charges[node_idx_lo + 1] += particle_spec.q*(particle_spec.xs[i] - node_list.Xs[node_idx_lo])
            end
        end

    elseif BC == "sheath"
        for particle_spec in particle_list
            for i = 1:length(particle_spec.xs)
                if particle_spec.xs[i] <= 0
                    node_list.charges[1] += particle_spec.q
                elseif particle_spec.xs[i] >= node_list.Xs[end]
                    node_list.charges[end] += particle_spec.q
                else
                    node_idx_lo = floor(Int64, particle_spec.xs[i]) + 1

                    node_list.charges[node_idx_lo] += particle_spec.q*(node_list.Xs[node_idx_lo + 1] - particle_spec.xs[i])
                    node_list.charges[node_idx_lo + 1] += particle_spec.q*(particle_spec.xs[i] - node_list.Xs[node_idx_lo])
                end
            end
        end

    elseif BC == "periodic"
        for particle_spec in particle_list
            for i = 1:length(particle_spec.xs)
                node_idx_lo = mod(floor(Int64, particle_spec.xs[i]), N_nodes)+1
                node_idx_hi = mod(node_idx_lo, N_nodes)+1

                node_list.charges[node_idx_lo] += particle_spec.q * (node_idx_lo - particle_spec.xs[i])
                node_list.charges[node_idx_hi] += particle_spec.q * (1 - (node_idx_lo - particle_spec.xs[i]))
                
            end
        end

    end
end

#= BIRDSALL & LANGDON EQ 2-5 (5-6) =#
function update_node_phi!(particle_list, node_list, N_nodes, dx, BC)
    A = zeros(N_nodes, N_nodes)
    for i = 1:N_nodes
        A[i, i] = -2
        A[mod(i, N_nodes)+1, i] = 1
        A[i, mod(i, N_nodes)+1] = 1
    end
    if BC == "zero" || BC == "sheath"
        A = Tridiagonal(A)
    elseif BC == "periodic"
        A = Symmetric(A)
    end
    node_list.phis = A \ (-node_list.charges/2)
end

#= BIRDSALL & LANGDON EQ 2-5 (4) =#
function update_node_E!(node_list, N_nodes, BC)
    M = zeros(N_nodes, N_nodes)
    for i = 1:N_nodes
        M[mod(i, N_nodes)+1, i] = 0.5
        M[i, mod(i, N_nodes)+1] = -0.5
    end
    if BC == "zero" || BC == "sheath"
        M = Tridiagonal(M)
        M[1, 1] = 1
        M[1, 2] = -1
        M[N_nodes, N_nodes-1] = 1
        M[N_nodes, N_nodes] = -1
    end
    node_list.Es = M*node_list.phis
end

#= BIRDSALL & LANGDON EQ 2-6 (3) =#
function update_particle_Es_phis!(particle_list, node_list, N_nodes, BC)
    if BC == "zero" || BC == "sheath"
        for particle_spec in particle_list
            for i = 1:length(particle_spec.xs)
                if particle_spec.xs[i] < 0 || particle_spec.xs[i] > node_list.Xs[end]
                    particle_spec.Es[i] = 0
                    particle_spec.phis[i] = 0
                    continue
                end
                node_idx_lo = floor(Int64, particle_spec.xs[i]) + 1
                
                particle_spec.Es[i] = (node_list.Xs[node_idx_lo + 1] - particle_spec.xs[i]) * node_list.Es[node_idx_lo] + (particle_spec.xs[i] - node_list.Xs[node_idx_lo]) * node_list.Es[node_idx_lo + 1]

                particle_spec.phis[i] = (node_list.Xs[node_idx_lo + 1] - particle_spec.xs[i]) * node_list.phis[node_idx_lo] + (particle_spec.xs[i] - node_list.Xs[node_idx_lo]) * node_list.phis[node_idx_lo + 1]
            end
        end

    elseif BC == "periodic"
        for particle_spec in particle_list
            for i = 1:length(particle_spec.xs)
                node_idx_lo = mod(floor(Int64, particle_spec.xs[i]), N_nodes)+1
                node_idx_hi = mod(node_idx_lo, N_nodes)+1

                particle_spec.Es[i] = (node_idx_lo - particle_spec.xs[i]) * node_list.Es[node_idx_lo] + (1 - (node_idx_lo - particle_spec.xs[i])) * node_list.Es[node_idx_hi]

                particle_spec.phis[i] = (node_idx_lo - particle_spec.xs[i]) * node_list.phis[node_idx_lo] + (1 - (node_idx_lo - particle_spec.xs[i])) * node_list.phis[node_idx_hi]
            end
        end

    end
end

#= BIRDSALL & LANGDON EQ 3-5 (3) =#
function update_particle_vs!(particle_list, dt)
    for particle_spec in particle_list
        particle_spec.vs_old = particle_spec.vs
        particle_spec.vs .+= particle_spec.q/particle_spec.m*particle_spec.Es*dt
    end
end

#= BIRDSALL & LANGDON EQ 3-5 (4) =#
function update_particle_xs!(particle_list, node_list, N_nodes, BC)
    if BC == "periodic"
        for particle_spec in particle_list
            particle_spec.xs = mod.(particle_spec.xs .+ particle_spec.vs, N_nodes)
        end

    else
        for particle_spec in particle_list
            particle_spec.xs .+= particle_spec.vs
        end

    end
end

#= BIRDSALL & LANGDON 3-10 =#
function init_particle_vs!(particle_list, node_list, N_nodes, BC, dx, dt)
    update_node_charge!(particle_list, node_list, N_nodes, BC)
    update_node_phi!(particle_list, node_list, N_nodes, dx, BC)
    update_node_E!(node_list, N_nodes, BC)
    update_particle_Es_phis!(particle_list, node_list, N_nodes, BC)
    for particle_spec in particle_list
        particle_spec.vs .+= (-particle_spec.q/2)/particle_spec.m*particle_spec.Es*dt
    end
end

function run_PIC(;
    names = ["electrons", "protons"], 
    qs = [-1, 1],
    ms = [1, mpme],
    # ms = [1, 10],
    temps = [0.005, 0.00], # eV
    drifts = [0.0, 0.0], # m/s
    n = [10^8, 10^8], #10^8
    NPs = [2000, 2000], 
    L_sys = 0.5, # m
    dt = 1e-8, # s
    BC = "periodic", 
    # BC = "zero",
    # BC = "sheath",
    # pos_IC = "uniform_exact",
    pos_IC = "uniform_exact_shuffle",
    # pos_IC = "uniform_rand",
    N_steps_max = 10000, 
    N_steps_save = 100, 
    plotting = false
    )

    # particle_list, N_species, node_list, N_nodes, dx, dt = init_PIC(BC = BC)
    
    #=
    ultra low Density (just reduce particle count)
    ultra high Density
    induce electric field
    add charge clumping
    plot distribution fxn after generation
    make linear fit of energy and plot for each sim
    SPURIOUS ELECTRIC FIELDS
    =#

    N_nodes = round(Int64, maximum(NPs)/10) # 10 particles per cell
    node_list = make_nodes(N_nodes)

    N_real_per_macro = n ./ NPs * L_sys
    qs .*= N_real_per_macro
    ms .*= N_real_per_macro
    particle_list, N_species = make_particles(names, NPs, qs, ms)

    if BC == "zero" || BC == "sheath"
        dx = L_sys/(N_nodes-1)
    elseif BC == "periodic"
        dx = L_sys/N_nodes
    end

    vel_dist_variance = zeros(N_species)
    for k = 1:N_species
        if pos_IC == "uniform_rand"
            particle_list[k].xs = rand(Uniform(0, L_sys/dx), NPs[k])
        elseif pos_IC=="uniform_exact" || pos_IC=="uniform_exact_shuffle"
            particle_list[k].xs = [L_sys/(NPs[k]*dx)*(i + k/N_species) for i = 0:NPs[k]-1]
            if pos_IC == "uniform_exact_shuffle"
                particle_list[k].xs .*= (1 .+.01*(rand(RandomDevice(), NPs[k]).-0.5))
            end
        end
        particle_list[k].xs = mod.(particle_list[k].xs, N_nodes) # start in domain regardless of boundary

        vel_dist_variance[k] = sqrt(temps[k]*e / (me*particle_list[k].m))
        # particle_list[k].vs = (rand(Normal(0, sqrt(temps[k]*e / (me*particle_list[k].m/N_real_per_macro[k])) ), NPs[k]) .+ drifts[k]) * dt/dx
        particle_list[k].vs = (rand(Normal(0.0, vel_dist_variance[k]), NPs[k]) .+ drifts[k]) * dt/dx

        shuffle!(RandomDevice(), particle_list[k].xs) # eliminate correlations
        shuffle!(RandomDevice(), particle_list[k].vs)
    end
    init_particle_vs!(particle_list, node_list, N_nodes, BC, dx, dt)

    #= SIMULATION STATISTICS =#
    debye_real = sqrt(eps0*temps[1]/(n[1]*e))
    debye_macro = sqrt(eps0*temps[1]/(length(particle_list[1].xs)/L_sys*e*abs(particle_list[1].q)))
    wp_real = sqrt(n[1]*e^2/(eps0*me))
    wp_macro = sqrt(length(particle_list[1].xs)/L_sys*(e*particle_list[1].q)^2/(eps0*me*particle_list[1].m))

    inputs_string = 
    """
                      Particle types: $(names)
                    Particle charges: $(qs./N_real_per_macro) e
                     Particle masses: $(ms./N_real_per_macro) me
                        Temperatures: $(temps) eV
                    Drift Velocities: $(drifts) m/s
            Number of macroparticles: $(NPs) 
                    Number Density n: $(n)
    Real particles per macroparticle: $(N_real_per_macro) 
                          Node Count: $(N_nodes)
                         System size: $(L_sys) m
                Grid node spacing dx: $(dx) m
                   Real Debye Length: $(debye_real) m
                  Macro Debye Length: $(debye_macro) m
            dx per Real Debye Length: $(debye_real/dx)
               Real Plasma Frequency: $(wp_real)
              Macro Plasma Frequency: $(wp_macro)
                        Time step dt: $(dt)
     Time steps / plasma oscillation: $(1/(dt*wp_real))
      Number of time steps simulated: $(N_steps_max)
                  Boundary Condition: $(BC)
          Initial Position Condition: $(pos_IC)
    """
    println(inputs_string)
    savefile = open("PIC Output/_Simulation_Info.txt", "w")
    write(savefile, inputs_string)

    #= ENERGY CONSERVATION TRACKING =#
    KE_spec = [Array{Float64, 1}(undef, N_steps_max) for i=1:N_species]
    PE_spec = [Array{Float64, 1}(undef, N_steps_max) for i=1:N_species]
    TE_spec = [Array{Float64, 1}(undef, N_steps_max) for i=1:N_species]
    KEs = Array{Float64, 1}(undef, N_steps_max)
    PEs = Array{Float64, 1}(undef, N_steps_max)
    TEs = Array{Float64, 1}(undef, N_steps_max)
    # PE_alt = Array{Float64, 1}(undef, N_steps_max) # from grid nodes

    #= MOVING AVERAGES =#
    moving_avg_x_log = [zeros(length(particle_list[i].xs)) for i=1:N_species]
    moving_avg_v_log = [zeros(length(particle_list[i].xs)) for i=1:N_species]
    moving_avg_phi_log = zeros(N_nodes)
    moving_avg_E_log = zeros(N_nodes)
    # N_steps_avg = N_steps_save/10
    N_steps_avg = 10

    #= VELOCITY DISTRIBUTION TRACKING =#
    # mean, variance, mean error, variance error
    vel_dist_params = [[[] for j = 1:4] for i=1:N_species]

    #= PARTICLE POSITION TRACKING FOR FFT =#
    particle_pos = [Array{Float64, 1}(undef, N_steps_max) for i=1:NPs[1]]
    fft_freqs = [i*1/dt/N_steps_max for i = 0:N_steps_max-1]

    step_idx = 0
    fig_idx = 3
    println("Progress:")
    #= MAIN SIMULATION LOOP =#
    while step_idx < N_steps_max
        update_node_charge!(particle_list, node_list, N_nodes, BC)
        update_node_phi!(particle_list, node_list, N_nodes, dx, BC)
        update_node_E!(node_list, N_nodes, BC)
        update_particle_Es_phis!(particle_list, node_list, N_nodes, BC)
        update_particle_vs!(particle_list, dt)
        update_particle_xs!(particle_list, node_list, N_nodes, BC)
        step_idx += 1
        if mod(step_idx, round(min(N_steps_max/10, 100))) == 0 # TRACK PROGRESS
            print("\e[2K") # clear line
            print("\e[1G") # move cursor to first column
            print(step_idx," / ",N_steps_max)
            if step_idx == N_steps_max
                print("\n")
            end
        end

        for i = 1:NPs[1]
            particle_pos[i][step_idx] = particle_list[1].xs[i] * dx
        end

        #= ENERGY CONSERVATION TRACKING =#
        for k in 1:N_species
            #= Kinetic Energy m*Vold*Vnew/2 =#
            KE_spec[k][step_idx] = sum(0.5 * dx/dt * dx/dt * me * particle_list[k].vs .* particle_list[k].vs_old * particle_list[k].m)
            #= Potential Energy qΦ =#
            PE_spec[k][step_idx] = sum(particle_list[k].phis * e/(eps0*dx) * particle_list[k].q * e)
            #= Total Energy =#
            TE_spec[k][step_idx] = KE_spec[k][step_idx] + PE_spec[k][step_idx]

            #= MOVING AVERAGES FOR PLOTTING =#
            if mod(step_idx, N_steps_save) >= N_steps_save-N_steps_avg
                moving_avg_x_log[k] .+= particle_list[k].xs * dx
                moving_avg_v_log[k] .+= particle_list[k].vs * dx/dt
            end

            #= VELOCITY DISTRIBUTION =#
            if mod(step_idx, N_steps_save) == 0
                vel_dist = params(fit(Normal, moving_avg_v_log[k]/N_steps_avg))
                push!(vel_dist_params[k][1], vel_dist[1])
                push!(vel_dist_params[k][2], vel_dist[2])
                push!(vel_dist_params[k][3], (vel_dist[1])/vel_dist_variance[k])
                push!(vel_dist_params[k][4], (vel_dist[2] - vel_dist_variance[k])/vel_dist_variance[k])
            end
        end
        KEs[step_idx] = sum([KE_spec[i][step_idx] for i=1:N_species])
        PEs[step_idx] = sum([PE_spec[i][step_idx] for i=1:N_species])
        TEs[step_idx] = sum([TE_spec[i][step_idx] for i=1:N_species])
        # PE_alt[step_idx] = sum(node_list.phis * e/(eps0*dx) * node.charge * e)

        #= MOVING AVERAGES FOR PLOTTING =#
        if mod(step_idx, N_steps_save) >= N_steps_save-N_steps_avg
            moving_avg_phi_log .+= node_list.phis * e/(eps0*dx)
            moving_avg_E_log .+= node_list.Es * e/(eps0*dx^2)
        end

        #= PHASE SPACE & NODE PLOTS =#
        if plotting == true || step_idx == 1
            if step_idx == 1 || mod(step_idx, N_steps_save) == 0
                fig_idx += 1
                # fig1 = figure(fig_idx)
                fig1 = figure(0)
                fig1.clf()
                (ax1, ax2, ax3, ax4) = fig1.subplots(nrows = 2, ncols = 2)
                # ax1.set_xlabel("Particle Position, m")
                ax1.set_ylabel("Particle Velocity, m/s", fontsize=8)
                ax2.set_xlabel("Particle Position, m", fontsize=8)
                ax1.set_xlim((node_list.Xs[1]*dx, (node_list.Xs[end]+1)*dx))
                ax2.set_xlim((node_list.Xs[1]*dx, (node_list.Xs[end]+1)*dx))
                ax2.set_ylabel("Particle Velocity, m/s", fontsize=8)
                # ax3.set_xlabel("Node Position, m")
                ax3.set_ylabel("Node Potential, V", fontsize=8)
                ax4.set_xlabel("Node Position, m", fontsize=8)
                ax4.set_ylabel("Node Electric Field, V/m", fontsize=8)
                # Plot phase space representation of particles
                if step_idx == 1
                    ax1.scatter(particle_list[1].xs * dx, 
                                particle_list[1].vs * dx/dt)
                    ax2.scatter(particle_list[2].xs * dx, 
                                particle_list[2].vs * dx/dt)
                else
                    moving_avg_x_log /= N_steps_avg
                    moving_avg_v_log /= N_steps_avg
                    ax1.scatter(moving_avg_x_log[1], moving_avg_v_log[1])
                    ax2.scatter(moving_avg_x_log[2], moving_avg_v_log[2])
                end
                # Plot grid potential & field
                if step_idx == 1
                    node_phis = node_list.phis * e/(eps0*dx)
                    node_Es = node_list.Es * e/(eps0*dx^2)
                    ax3.plot(node_list.Xs, node_phis)
                    ax4.plot(node_list.Xs, node_Es)
                else
                    moving_avg_phi_log /= N_steps_avg
                    moving_avg_E_log /= N_steps_avg
                    ax3.plot(node_list.Xs, moving_avg_phi_log)
                    ax4.plot(node_list.Xs, moving_avg_E_log)
                end
                fig1.set_size_inches(10, 8)
                fig1.tight_layout()
                fig1.savefig("PIC Output/fig_$(fig_idx-3).png", dpi=50)
                # close(fig1)

                moving_avg_x_log = [zeros(length(particle_list[i].xs)) for i=1:N_species]
                moving_avg_v_log = [zeros(length(particle_list[i].xs)) for i=1:N_species]
                moving_avg_phi_log = zeros(N_nodes)
                moving_avg_E_log = zeros(N_nodes)
            end
        end
    end

    times = [i*dt for i = 1:N_steps_max]

    #= TOTAL ENERGY LINEAR FIT =#
    TE_DF = DataFrame(times = times, TEs = TEs)
    TE_lm = lm(@formula(TEs ~ times), TE_DF)
    TE_coeffs = coef(TE_lm)
    TE_fit = TE_coeffs[1] .+ TE_coeffs[2]*times

    #= ENERGY CONSERVATION PLOTS =#
    fig1 = figure(0)
    fig1.clf()
    fig1.set_size_inches(12, 8)
    fig2 = figure(1)
    fig2.clf()
    fig2.set_size_inches(12, 8)
    fig3 = figure(2)
    fig3.clf()
    fig3.set_size_inches(12, 8)
    (ax1, ax2, ax3) = fig1.subplots(nrows = 3, ncols = 1)
    (ax4, ax5, ax6) = fig2.subplots(nrows = 3, ncols = 1)
    (ax7, ax8, ax9) = fig3.subplots(nrows = 3, ncols = 1)
    fig1.suptitle("Electron Energy Conservation",fontsize=14,fontweight="bold")
    ax1.set_title("Kinetic Energy, J")
    ax2.set_title("Potential Energy, J")
    ax3.set_title("Total Energy, J")
    ax3.set_xlabel("Time, s")
    fig2.suptitle("Proton Energy Conservation",fontsize=14,fontweight="bold")
    ax4.set_title("Kinetic Energy, J")
    ax5.set_title("Potential Energy, J")
    ax6.set_title("Total Energy, J")
    ax6.set_xlabel("Time, s")
    fig3.suptitle("Total Energy Conservation",fontsize=14,fontweight="bold")
    ax7.set_title("Kinetic Energy, J")
    ax8.set_title("Electric Potential Energy, J")
    ax9.set_title("Total Energy, J")
    ax9.set_xlabel("Time, s")
    ax1.plot(times, KE_spec[1],
        color = (0,0,0), 
        linestyle = "-", 
        label = particle_list[1].name)
    ax2.plot(times, PE_spec[1],
        color = (0,0,0), 
        linestyle = "-", 
        label = particle_list[1].name)
    ax3.plot(times, TE_spec[1],
        color = (0,0,0), 
        linestyle = "-", 
        label = particle_list[1].name)
    ax4.plot(times, KE_spec[2],
        color = (0,0,0), 
        linestyle = "-", 
        label = particle_list[2].name)
    ax5.plot(times, PE_spec[2],
        color = (0,0,0), 
        linestyle = "-", 
        label = particle_list[2].name)
    ax6.plot(times, TE_spec[2],
        color = (0,0,0), 
        linestyle = "-", 
        label = particle_list[2].name)
    ax7.plot(times, KEs,
        color = (0,0,0),
        linestyle = "-")
    ax8.plot(times, PEs,
        color = (0,0,0), 
        linestyle = "-")
    # ax8.plot(times, PE_alt,
    #     color = (0.5,0.5,0.5), 
    #     linestyle = "-")
    ax9.plot(times, TEs,
        color = (0,0,0), 
        linestyle = "-")
    ax9.plot(times, TE_fit, # TE LINEAR FIT
        color = (0.5,0.5,0.5), 
        linestyle = ":")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig("PIC Output/Energy Conservation Electrons.png", dpi=100)
    fig2.savefig("PIC Output/Energy Conservation Protons.png", dpi=100)
    fig3.savefig("PIC Output/Energy Conservation Total.png", dpi=100)
    close(fig1)
    close(fig2)
    close(fig3)

    max_TE = max(maximum(TEs)-TEs[1], TEs[1]-minimum(TEs))/TEs[1]*100
    RMS_TE = sum(((TEs.-TEs[1])/TEs[1]).^2/N_steps_max)*100
    RMS_vel_mu_err = zeros(N_species)
    RMS_vel_var_err = zeros(N_species)
    mean_vel_mu_err = zeros(N_species)
    mean_vel_var_err = zeros(N_species)
    for k in 1:N_species
        RMS_vel_mu_err[k] = sqrt(sum(vel_dist_params[k][3].^2)/length(vel_dist_params[k][3]))
        RMS_vel_var_err[k] = sqrt(sum(vel_dist_params[k][4].^2)/length(vel_dist_params[k][4]))
        mean_vel_mu_err[k] = sum(vel_dist_params[k][3])/length(vel_dist_params[k][3])
        mean_vel_var_err[k] = sum(vel_dist_params[k][4])/length(vel_dist_params[k][4])
    end
    diagnostics_string = 
    """
    
    ENERGY CONSERVATION
    Max Total Energy Error: $(max_TE) %
    RMS Total Energy Error: $(RMS_TE) %
    Last Step Total Energy Error: $((TEs[end]-TEs[1])/TEs[1]*100) %
    Total Energy Linear Fit Slope: $(TE_coeffs[2]/TEs[1]*100*dt) %/dt

    VELOCITY DISTRIBUTION
    RMS Mean Error: $(RMS_vel_mu_err*100) %
    RMS Variance Error: $(RMS_vel_var_err*100) %
    Mean Mean Error: $(mean_vel_mu_err*100) %
    Mean Variance Error: $(mean_vel_var_err*100) %
    First Step Mean Error: $([vel_dist_params[1][3][1]*100, vel_dist_params[2][3][1]*100]) %
    First Step Variance Error: $([vel_dist_params[1][3][1]*100, vel_dist_params[2][3][1]*100]) %
    Last Step Mean Error: $([vel_dist_params[1][3][end]*100, vel_dist_params[2][3][end]*100]) %
    Last Step Variance Error: $([vel_dist_params[1][3][end]*100, vel_dist_params[2][3][end]*100]) %
    """
    println(diagnostics_string)
    write(savefile, diagnostics_string)
    close(savefile)

    particle_fft = [Array{Float64, 1}(undef, N_steps_max) for i=1:NPs[1]]
    for i = 1:NPs[1]
        particle_fft[i] = abs.(fft(particle_pos[i]))
    end

    return TEs, particle_fft, fft_freqs
end