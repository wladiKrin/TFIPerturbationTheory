path = "/local/krinitsin"
# path = "/Users/wladi/Projects"

include(path*"/TFIPerturbationTheory/src/PertTheory.jl")


let
    L = 10 # lattice size

    J = -1;

    dt = 1e-3
    T = 100.

    gs      = [-0.5,-1.5]
    Ns      = [200]
    num_MCs = [50000,100000, 500000]

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
    @show prefactor * mean(S .* samples .* signs)

    Threads.@threads for (sample, sign) in collect(zip(Iterators.partition(samples, L), Iterators.partition(signs, L)))
        t = 0.
        dataTemp = []

        n0 = S .* sample

        @show S, prefactor * mean(sign .* n0)

        phi0 = 2*pi*rand(L)
        fields = (n0, phi0)

        # Compute time evolution
        try
            while t < T
                push!(dataTemp, [t, prefactor * mean(sign .* (S .- fields[1])), prefactor * mean(sign .* abs.(S .- fields[1])), prefactor * mean(sign .* abs2.(S .- fields[1]))])
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
    println("finished run")

    # df_res = analyze_data(data, params)
    name = path * "/TFIPerturbationTheory/data/TWA_SG22_L=$(L)_Sz=$(S)_num_MC=$(num_MC)_g=$(g)"

    h5open(name*".h5", "w") do file    

        file["prefactor"] = [prefactor]
        file["signs"] = permutedims(hcat(collect(Iterators.partition(signs, L))...))

        for (run, d) in enumerate(data)
            file["$(run)/time"]    = [dd[1] for dd in d]
            file["$(run)/meanSz"]  = [dd[2] for dd in d]
            file["$(run)/absSz"]   = [dd[3] for dd in d]
            file["$(run)/meanSz2"] = [dd[4] for dd in d]
        end
    end

    df = analyze_data3(path, (g,N,num_MC,L))
    CSV.write(name *".csv", df)

    println("finished analysis")
end;
