include("../src/PertTheory.jl")
using Statistics
#using Plots

gs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,-0.75,-0.8,-0.85,-0.9,-0.95,-1.0,-2.0]
gs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.75,-0.8,-0.85,-0.9,-0.95,-1.0,-1.25,-1.5,-1.75,-2.0]

## domain wall length operator;;; not very efficient just use H0 for that
function domainWallL(spins::Tuple{Vararg{Int64}}, L, neigh)
    spins = reshape([s for s in spins], L)

    D = 0
    for (i,j) in neigh
        pos1 = _coordinate_simple_lattice(i, L)
        pos2 = _coordinate_simple_lattice(j, L)
        s1 = spins[pos1[1],pos1[2]] == 1 ? 1 : -1
        s2 = spins[pos2[1],pos2[2]] == 1 ? 1 : -1
        D += (1-s1*s2)/2
    end
    return trunc(Int,D)
end;

## imbalance
function corrF(spins::Tuple{Vararg{Int64}}, L)
    spins = reshape([s for s in spins], L)

    left = mapreduce(+, Iterators.product(1:L[1], 1:floor(Int, L[2]/2))) do (i,j)
        spin = (spins[i,j] == 1) ? 1 : -1
        return spin
    end/(prod(L)/2)
    right = mapreduce(+, Iterators.product(1:L[1], floor(Int, L[2]/2)+1:L[2])) do (i,j)
        spin = (spins[i,j] == 1) ? 1 : -1
        return spin
    end/(prod(L)/2)
    return left*right
end

L= (4,4)
N = prod(L);
J = -1;
h = -0.
# g = -0.1

next_neighbours = nearest_neighbours(L, collect(1:prod(L)), periodic_y=false)

spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
dw_precalc = map(spin_basis) do spin
    return domainWallL(spin, L, next_neighbours)
end

### sort basis according to domain wall length ###
sorted_spin_basis = sort(collect(zip(dw_precalc, spin_basis)), by = x->x[1])
dw_precalc  = [d[1] for d in sorted_spin_basis]
spin_basis  = [d[2] for d in sorted_spin_basis]

# imb_precalc = map(s -> imbalance(s, L), spin_basis)
corr_precalc = map(s -> corrF(s, L), spin_basis)

spin_basis_table = Dict(
    map(enumerate(zip(spin_basis,dw_precalc))) do (i, (spin, dw))
        return (spin, (i,dw))
    end
);

### initial state ###
init_spin = vcat(fill(1,Int(N/2)),fill(0,Int(N/2)))
init_idx = first(spin_basis_table[Tuple(init_spin)]);

for g in gs

    @show g

    vals, vecs, dw = readSpec("../data/spec_ED_Bound_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

    #= 
    corr = map(1:2^16) do i
        v = vecs[:,i]
        return sum(abs2.(vecs[:,i]) .* corr_precalc)
    end

    df = DataFrame(vals = vals, dw = dw, occ = vecs[init_idx,:], corr = corr)
    CSV.write("../data/occSpecB_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
end
    =#

    isSort = sort(collect(enumerate(zip(abs2.(vecs[init_idx,:]),dw))), by=x->x[2], rev=true)

    is = []
    occs = []
    occ = 0

    for (i,(o,dw)) in isSort
        occ += o
        push!(is, (i,o,dw))
        push!(occs, occ)

        occ >=0.99 && break
    end

    is = sort(is, by=x->x[3])

    @show length(is)

    iss   = [i[1] for i in is]
    occss = [i[2] for i in is]
    dws   = [i[3] for i in is]
    valss   = [vals[i] for i in iss]

    vecsWeights = [vecs[:,i] for i in iss]

    vecsWeights = permutedims(hcat(vecsWeights...))

    h5open("../data/vecs_vals_Bound_g=$(g).h5", "w") do file    
        file["is"] = iss
        file["vals"] = valss
        file["dw"] = dws
        file["occ"] = occss
        file["weight"] = vecsWeights
    end
end
#=
    
    hmData = map(Iterators.product(iss,iss)) do (i,j)
        return mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * imb_precalc[num]
        end
    end

    heatmap(title = "number of eigenstates = $(length(is))", hmData)

    hline!([l8], c = :red, label = "")
    vline!([l8], c = :red, label = "")

    hline!([l10], c = :blue, label = "")
    vline!([l10], c = :blue, label = "")

    savefig("../figures/imbHeatmap_timescale_g=$(g).png")

    hmDataFact = map(Iterators.product(iss,iss)) do (i,j)
        fact = conj(vecs[init_idx,i])*vecs[init_idx,j]
        return fact*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * imb_precalc[num]
        end
    end

    enDiff = map(Iterators.product(iss,iss)) do (i,j)
        return vals[j]-vals[i]
    end

    heatmap(title = "number of eigenstates = $(length(is))", hmDataFact)

    hline!([l8], c = :red, label = "")
    vline!([l8], c = :red, label = "")

    hline!([l10], c = :blue, label = "")
    vline!([l10], c = :blue, label = "")
    savefig("../figures/imbHeatmap_factor_g=$(g).png")

    h5open("../data/imbalanceData_hm_g=$(g).h5", "w") do file    
        file["imb"] =   hmData
        file["imbLog"] = hmDataFact
        file["enDiff"] = enDiff
    end


    obs8 = []
    obs810 = []
    obs10 = []
    obsElse = []

    for (i,j) in Iterators.product(is,is)
        fact = conj(vecs[init_idx,i])*vecs[init_idx,j]
        res = [abs(fact*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * imb_precalc[num]
        end), abs(vals[i]-vals[j])]
        if (dw[i]>7 && dw[i]<9) && (dw[j]>7 && dw[j]<9)
            push!(obs8, res)
        elseif (dw[i]>9 && dw[i]<11) && (dw[j]>9 && dw[j]<11)
            push!(obs10, res)
        elseif (dw[i]>7 && dw[i]<9) && (dw[j]>9 && dw[j]<11)
            push!(obs810, res)
        elseif (dw[i]>9 && dw[i]<11) && (dw[j]>7 && dw[j]<9)
            push!(obs810, res)
        else
            push!(obsElse, res)
        end
    end

    h5open("../data/imbalanceData_g=$(g).h5", "w") do file    
        file["imb8"] =    [o[1] for o in obs8]
        file["enDiff8"] = [o[2] for o in obs8]
        file["imb10"] =    [o[1] for o in obs10]
        file["enDiff10"] = [o[2] for o in obs10]
        file["imb810"] =    [o[1] for o in obs810]
        file["enDiff810"] = [o[2] for o in obs810]
        if !isempty(obsElse)
            file["imbElse"] =    [o[1] for o in obsElse]
            file["enDiffElse"] = [o[2] for o in obsElse]
        end
    end

    scatter(xtitle = "number of eigenstates = $(length(is)), norm = $(occ)", label="imbalance matrix element", ylabel="energy difference",
        dpi = 300,
    )

    scatter!([o[1] for o in obs8], [o[2] for o in obs8],     markershape = :xcross,  label = "dw=8")
    scatter!([o[1] for o in obs10], [o[2] for o in obs10],   markershape = :rect,    label = "dw=10")
    scatter!([o[1] for o in obs810], [o[2] for o in obs810], markershape = :diamond, label = "8 cross 10")
    scatter!([o[1] for o in obsElse], [o[2] for o in obsElse], label = "else")
    savefig("../figures/imbEnergy_factor_lin_g=$(g).png")


    scatter(title = "number of eigenstates = $(length(is)), norm = $(occ)", xlabel="imbalance matrix element", ylabel="energy difference",
        yaxis = :log, ylim = (minLim,10),
        dpi = 300,
    )

    scatter!([o[1] for o in obs8], [o[2] for o in obs8],     markershape = :xcross,  label = "dw=8")
    scatter!([o[1] for o in obs10], [o[2] for o in obs10],   markershape = :rect,    label = "dw=10")
    scatter!([o[1] for o in obs810], [o[2] for o in obs810], markershape = :diamond, label = "8 cross 10")
    scatter!([o[1] for o in obsElse], [o[2] for o in obsElse], label = "else")
    savefig("../figures/imbEnergy_factor_log_g=$(g).png")


    H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
    N = sparse(diagm(fill(16, 2^16)) + H0 ./ 2)
    xCorrM  = build_twoCorrel(spin_basis, spin_basis_table)
    xPolM       = build_xPol(spin_basis, spin_basis_table)

    is = [195,196,197,198,199,200,201,202]
    is2 = collect(99:522)
    @show mean(map(i->dw[i], is))
    @show std(map(i->dw[i], is))
    @show mean(map(i->dw[i], is2))
    @show std(map(i->dw[i], is2))

    #occ = abs2.(vecs[476,:])

    occSum = sum(map(x->abs2(vecs[476,x]), is))
    occSum2 = sum(map(x->abs2(vecs[476,x]), is2))
    println("occ1 = ", occSum)
    println("occ2 = ", occSum2)
    dw1 = sum(map(x->dw[x] * abs2(vecs[476,x]), is))
    dw2 = sum(map(x->dw[x] * abs2(vecs[476,x]), is2))
    @show dw1/occSum
    @show dw2/occSum


    is   = [i for (i,dw) in enumerate(dw) if dw > 7 && dw<9]
    is10 = [i for (i,dw) in enumerate(dw) if dw > 9 && dw<11]

    println("built states")

    ### only dw=8 states ###

    println("1")
    obs1 = map(Iterators.product(is,is)) do (i,j)
        imb = conj(vecs[init_idx,i])*vecs[init_idx,j]*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * imb_precalc[num]
        end

        dw = conj(vecs[init_idx,i])*vecs[init_idx,j]*dot(vecs[:,i],N*vecs[:,j])

        xPol = conj(vecs[init_idx,i])*vecs[init_idx,j]*dot(vecs[:,i],xPolM*vecs[:,j])

        xCorr = conj(vecs[init_idx,i])*vecs[init_idx,j]*dot(vecs[:,i],xCorrM*vecs[:,j])

        return ((i,j), imb, dw, xPol, xCorr)
    end

    data1 = map(ts) do t
        obs = real(mapreduce(+, obs1) do ((i,j), o, d, xP, xC)
            return exp(-1im*t*(vals[j]-vals[i])) .* [o, d, xP, xC]
        end)

        (imb, dw, xPol, xCorr) = obs

        return [t, imb, dw, xPol, xCorr]
    end

    df1 = DataFrame(t = [d[1] for d in data1], imb = [d[2] for d in data1], dw = [d[3] for d in data1], xPol = [d[4] for d in data1], xCorr = [d[5] for d in data1])
    CSV.write("../data/timeEv_dw8states_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df1)

    ### 8 & 10 states ###
    println("8 and 10")
    is = vcat(is, is10)
    @show length(is)
    @time "built obs" obs2 = map(Iterators.product(is,is)) do (i,j)
        @time begin
            fact = conj(vecs[init_idx,i])*vecs[init_idx,j]
            imb = fact*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
                return conj(v1) * v2 * imb_precalc[num]
            end

            dw = fact*dot(vecs[:,i],N*vecs[:,j])

            xPol = fact*dot(vecs[:,i],xPolM*vecs[:,j])

            xCorr = fact*dot(vecs[:,i],xCorrM*vecs[:,j])
        end

        return ((i,j), imb, dw, xPol, xCorr)
    end

    @time "time evol" data2 = map(ts) do t
        obs = real(mapreduce(+, obs2) do ((i,j), o, d, xP, xC)
            return exp(-1im*t*(vals[j]-vals[i])) .* [o, d, xP, xC]
        end)

        (imb, dw, xPol, xCorr) = obs

        return [t, imb, dw, xPol, xCorr]
    end

    df2 = DataFrame(t = [d[1] for d in data2], imb = [d[2] for d in data2], dw = [d[3] for d in data2], xPol = [d[4] for d in data2], xCorr = [d[5] for d in data2])
    CSV.write("../data/timeEv_dw8_10states_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df2)

    continue

    #is2 = [en[1] for en in filter(d -> abs(d[2]-8) < 1e-1, collect(enumerate(dw)))]

    #@show is2
    #@show map(i -> dw[i], is2)

    occ = abs2.(vecs[476,:])
    #occRed = [(i,round(o,digits=4)) for (i,o) in enumerate(occ) if o > 1e-3]
    occRed = collect(enumerate(occ))

    #println(sort(occRed, by=x->x[2], rev=true))
    #println(map(x->occ[x], is))
    occSum = sum(map(x->occ[x], is))
    occSum2 = mapreduce(+, occRed) do (i,o)
       return o
    end
    occSum3 = sum(map(x->occ[x], is2))
    println("occ1 = ", occSum)
    println("occ2 = ", occSum2)
    println("occ3 = ", occSum3)
    println("dw1 = ", sum(map(i->occ[i]*dw[i], is))/occSum)
    println("dw2 = ", sum(map(occRed) do (i,o)
                            return o * dw[i]
                        end)/occSum2)
    println("dw3 = ", sum(map(i->occ[i]*dw[i], is2))/occSum3)

    return 

    nums = sort(occred, by=x->x[2], rev=true) # map(x->occ[x], [195,196,197,198,199,200,201,202])
    cs = [1e-4,1e-3,5e-4,1e-3,1e-3,1e-3,1e-3,1e-3]

    occs = map(zip(nums, cs)) do (num, c)
        return scatter(title = "ev num=$(num[1]), overlap = $(round(num[2], digits=2))", xlabel = "spin basis", ylabel = "overlap", label = "",
            [(i,o) for (i,o) in enumerate(vecsF[:,num[1]]) if abs2(o)>1e-3])
    end
    hms = map(zip(nums, cs)) do (num, c)
        return heatmap(title = "ev num=$(num[1]), overlap = $(round(num[2], digits=2))", clim = (-1,1),
            mapreduce(+, [(i,o) for (i,o) in enumerate(vecsF[:,num[1]]) if abs2(o)>1e-3]) do (i,o)
                return abs2(o)*toSpinMatr(spin_basis[i])
            end)
    end
    plot(vcat(occs, hms)..., layout=(2,7), size = (4000,600))
    savefig("eiv_decomposition_IS_g=$(g).png")
end
=#
