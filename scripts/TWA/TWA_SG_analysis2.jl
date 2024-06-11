path = "/local/krinitsin"
# path = "/Users/wladi/Projects"

include(path*"/TFIPerturbationTheory/src/PertTheory.jl")

BLAS.set_num_threads(1)

gs      = [-0.5]#,-1.5]
Ns      = [10,50,100,200]
num_MCs = [1000,5000,10000]
for (g,N,num_MC) in collect(Iterators.product(gs,Ns,num_MCs))
    L = 10 # lattice size

    J = -1;

    dt = 1e-3
    T = 10.


    # params = collect(Iterators.product(gs, Ns, num_MCs))
    # g, N, num_MC = params[parse(Int, ARGS[1])]
    @show g, N, num_MC

    S = N/2 # initial spin value

    name = path * "/TFIPerturbationTheory/data/TWA_SG_L=$(L)_Sz=$(S)_num_MC=$(num_MC)_g=$(g)"

    h5open(name*".h5", "r") do file    

        prefactor = read(file, "prefactor")[1]
        signs = read(file, "signs")
        data = map(1:num_MC) do run
            time  = read(file, "$(run)/time")
            occ   = read(file, "$(run)/occ")
            return occ[1,:]
        end

        Sz = permutedims(hcat(data...))

        #Sz = map(1:length(data[1][1])) do i 
        #    return permutedims(hcat(map(data) do data_t
        #        return data_t[2][i,:]
        #    end...))
        #end
        # @show Sz

        @show typeof(Sz)
        @show size(Sz)
        @show size(Sz[1])
        @show typeof(signs)
        @show size(signs)

        @show mean(vec(Sz[1] .* signs))
        @show mean(vec(Sz))
        res = Sz .* signs
        @show size(res)
        @show size(vec(res))
        @show mean(vec(res))

        # Sz = map(data) do data_t
        #     return map(1:size(data_t[3])[1]) do i
        #         return data_t[3][i,:]
        #     end
        # end
        # signs = [d[2] for d in data]

    end
   #     meanSz   = [mean(vec(sz .* signs)) for sz in Sz]
   #     absSz    = [mean(vec(abs.(sz) .* signs)) for sz in Sz]
   #     meanSz2  = [mean(vec((sz.^2) .* signs)) for sz in Sz]
   #     # meanSz  = [sum([mean(sign .* s[i]) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]
   #     # meanSz2 = [sum([mean(sign .* (s[i].^2)) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]
   #     # absSz   = [sum([mean(sign .* abs.(s[i])) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]

   #     return DataFrame(
   #         t       = data[1][1],
   #         meanSz  = prefactor * meanSz,
   #         meanSz2 = prefactor * meanSz2,
   #         absSz   = prefactor * absSz,
   #     );
   # end
   # CSV.write(name *".csv", df)

    println("finished")
end;
