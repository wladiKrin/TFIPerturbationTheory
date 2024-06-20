path = "/local/krinitsin"
# path = "/Users/wladi/Projects"
# using CairoMakie

include(path*"/TFIPerturbationTheory/src/PertTheory.jl")


let
    L = 10 # lattice size

    J = -1;

    T = 100.

    gs      = [-0.5,-1.0,-1.5,-2.0]
    Ns      = [200]
    num_MCs = [10000] #, 500_000]

    params = collect(Iterators.product(gs, Ns, num_MCs))
    # g, N, num_MC = params[1] 
    g, N, num_MC = params[parse(Int, ARGS[1])]
    @show g, N, num_MC

    S = N/2 # initial spin value

    params = (J,g,S)
    obs = obs_SG2
    F   = F_SG3

    data = [[Vector{Float64}(undef, 10001) for _ in 1:4] for _ in 1:num_MC]

    # create samples
    samples, signs, prefactor = get_samples(N; n_samples = num_MC * L, dx=1e-4)
    @show prefactor * mean(S .* samples .* signs)

    Threads.@threads for (i,(sample, sign)) in collect(enumerate(collect(zip(Iterators.partition(samples, L), Iterators.partition(signs, L)))))
    # = 0; chunks = partition(1:1000000, Threads.nthreads());julia> tasks = map(chunks) do chunk           Threads.@spawn begin               x = 0               for i in chunk                   x += i               end               x           end       end;julia> thread_sums = fetch.(tasks);julia> for i in thread_sums           s += i       end
    # for (sample, sign) in collect(zip(Iterators.partition(samples, L), Iterators.partition(signs, L)))
        t = 0.
        dataTemp = []

        n0 = S .* sample

        # @show S, prefactor * mean(sign .* n0)

        phi0 = 2*pi*rand(L)
        fields = vcat(n0, phi0)
        
        prob = ODEProblem(F, fields, (0., T), params)
        sol = solve(prob, reltol=1e-6, abstol=1e-6, saveat=0.01)
        @show length(sol.t)

        # push!(data, [[time, prefactor * mean(sign .* (S .- n[1:L])), prefactor * mean(sign .* abs.(S .- n[1:L])), prefactor * mean(sign .* abs2.(S .- n[1:L]))] for (time, n) in zip(sol.t, sol.u)])
        data[i] = [[time, prefactor * mean(sign .* (S .- n[1:L])), prefactor * mean(sign .* abs.(S .- n[1:L])), prefactor * mean(sign .* abs2.(S .- n[1:L]))] for (time, n) in zip(sol.t, sol.u)]
    end
    sleep(30)
    println("finished run")

    # df_res = analyze_data(data, params)
    # @show df_res
    name = path * "/TFIPerturbationTheory/data/TWA_SG_phi_sin(phi)_L=$(L)_Sz=$(S)_num_MC=$(num_MC)_g=$(g)"

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

    df = analyze_data3(name, num_MC)
    # fig = Figure()
    # ax1 = Axis(fig[1, 1])
    # ax2 = Axis(fig[2, 1])
    # ax3 = Axis(fig[3, 1])
    #
    # lines!(ax1, df.t, df.meanSz, label = "meanSz")
    # lines!(ax2, df.t, df.absSz, label = "meanSz")
    # lines!(ax3, df.t, df.meanSz2, label = "meanSz")
    #
    # fig
    CSV.write(name *".csv", df)

    println("finished analysis")
end;
