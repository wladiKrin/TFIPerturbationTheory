path = "/local/krinitsin"
# path = "/Users/wladi/Projects"

include(path*"/TFIPerturbationTheory/src/PertTheory.jl")


let
    L = 10 # lattice size

    J = -1;

    dt = 1e-3
    T = 10.

    gs      = [-0.5,-1.5]
    Ns      = [10,50,100,200]
    num_MCs = [1000,5000,10000]

    params = collect(Iterators.product(gs, Ns, num_MCs))
    g, N, num_MC = params[parse(Int, ARGS[1])]
    @show g, N, num_MC

    S = N/2 # initial spin value

    params = (J,g,S)
    obs = obs_SG2
    F   = F_SG2

    data = []

    # create samples
    samples, signs, prefactor = get_samples(N; n_samples = num_MC * L)

    Threads.@threads for (sample, sign) in collect(zip(Iterators.partition(samples, L), Iterators.partition(signs, L)))
        t = 0.
        dataTemp = []

        n0 = (S/2) * sample
        phi0 = 2*pi*rand(L)
        fields = (n0, phi0)

        # Compute time evolution
        try
            while t < T
                push!(dataTemp, [t, obs_SG2(fields, params)])
                fields = heun_step(fields, params, F, dt)
                # dt_new, fields = adaptive_heun_step(fields, params, F, dt, 1e-1)
                t += dt
            end
        catch
            @warn "Error, fields = $fields"
            continue
        end
        push!(data, dataTemp)
    end

    # df_res = analyze_data(data, params)
    name = path * "/TFIPerturbationTheory/data/TWA_SG_L=$(L)_Sz=$(S)_num_MC=$(num_MC)_g=$(g)"

    h5open(name*".h5", "w") do file    

        file["prefactor"] = [prefactor]
        file["signs"] = permutedims(hcat(collect(Iterators.partition(signs, L))...))

        for (run, d) in enumerate(data)
            file["$(run)/time"] = [dd[1] for dd in d]
            file["$(run)/occ"]  = permutedims(hcat([dd[2] for dd in d]...))
        end
    end
    println("finished")
end;
