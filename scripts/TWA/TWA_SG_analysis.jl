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

    name = path * "/TFIPerturbationTheory/data/TWA_SG_L=$(L)_Sz=$(S)_num_MC=$(num_MC)_g=$(g)"
    df = analyze_data2(path, (g,N,num_MC,L))
    CSV.write(name *".csv", df)

    println("finished")
end;
