include("../src/SW.jl")

gs = [-1.0,-1.25,-1.5,-1.75,-2.0,-2.25,-2.5] #-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,
for g in gs
#let 
    L= (4,4)
    N = prod(L);
    J = -1;
    h = -0.
    #g = -0.5

    next_neighbours = nearest_neighbours(L, collect(1:N))
    #@show N
    #@show 2^BigInt(N)

    #@show collect(Iterators.product(fill([1,0],BigInt(N))...))

    @time spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
    @time dw_precalc = map(spin_basis) do spin
        return domainWallL(spin, L)
    end

    ### sort basis according to domain wall length ###
    @time sorted_spin_basis = sort(collect(zip(dw_precalc, spin_basis)), by = x->x[1])
    dw_precalc  = [d[1] for d in sorted_spin_basis]
    spin_basis  = [d[2] for d in sorted_spin_basis]

    @time imb_precalc = map(s -> imbalance(s, L), spin_basis)

    spin_basis_table = Dict(
        map(enumerate(zip(spin_basis,dw_precalc))) do (i, (spin, dw))
            return (spin, (i,dw))
        end
    );

    domainWall_table = Dict(
        map(enumerate(dw_precalc)) do (i,  dw)
            return (i, dw)
        end
    );

    ### initial state ###
    init_spin = vcat(fill(1,Int(N/2)),fill(0,Int(N/2)))
    init_idx = first(spin_basis_table[Tuple(init_spin)]);


    #=
    psi=zeros(length(spin_basis))
    psi[init_idx]=1
    psi = sparse(psi)

    ### build Hamiltonians ###
    H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
    H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
    V1 = H1+R1;

    #======================== full =============================#
    H_full = H0+V1

    #==================== perturbative =========================#
    vecH, vecR, vecS = SW_transformation(H0, H1, R1, domainWall_table, n_order)
    H_pert = dropzeros(reduce(+, vecH; init=H0))

    ### observables ###
    H0_corr = build_corr(H0, vecS)
    D0      = dropzeros(reduce(+, vecH))
    =#

    vals, vecs, dw = readSpec("../data/spec_ED_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

    is = [195,196,197,198,199,200,201,202]
    is2 = collect(99:522)
    #occ = abs2.(vecs[476,:])
    occSum = sum(map(x->abs2.(vecs[476,x]), is))
    occSum2 = sum(map(x->abs2.(vecs[476,x]), is2))
    println("occ1 = ", occSum)
    println("occ2 = ", occSum2)
    continue


    ts = [0.0,0.1]
    step = 1.1
    tmax = 1e20

    ### logarithmic timesteps ###
    while true
      push!(ts, ts[end]*step)
      ts[end] > tmax && break
    end

    @show g
    H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
    N = sparse((diagm(fill(32, 2^16)) + H0) ./ 2)
    display(H0)
    display(N)
    return

    #=
    i = 99+5
    j = 522-3
    @show (dot(vecs[:,i],H0*vecs[:,j]))
    @show mapreduce(+, enumerate(zip(Transpose(vecs[:,i]),vecs[:,j]))) do (num,(v1,v2))
        return conj(v1) * v2 * dw_precalc[num]
    end
    =#

    is = [195,196,197,198,199,200,201,202]
    is = collect(99:522)

    @show mapreduce(+, Iterators.product(is,is)) do (i,j)
        return conj(vecs[init_idx,i])*vecs[init_idx,j]*dot(vecs[:,i],N*vecs[:,j])    
    end

    @show 16 + mapreduce(+, Iterators.product(is,is)) do (i,j)
        return conj(vecs[init_idx,i])*vecs[init_idx,j]*dot(vecs[:,i],H0*vecs[:,j])    
    end/2

    @show mapreduce(+, Iterators.product(is,is)) do (i,j)
        return conj(vecs[init_idx,i])*vecs[init_idx,j]*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * dw_precalc[num]
        end
    end

    return

    is = [195,196,197,198,199,200,201,202]
    is2 = collect(99:522)
    occ = abs2.(vecs[476,:])
    occSum = sum(map(x->occ[x], is))
    occSum2 = sum(map(x->occ[x], is2))
    println("occ1 = ", occSum)
    println("occ2 = ", occSum2)
    #continue


    ### only 8 states ###
    println("1")
    is = [195,196,197,198,199,200,201,202]
    obs1 = map(Iterators.product(is,is)) do (i,j)
        imb = conj(vecs[init_idx,i])*vecs[init_idx,j]*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * imb_precalc[num]
        end
        dw = conj(vecs[init_idx,i])*vecs[init_idx,j]*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * dw_precalc[num]
        end
        return ((i,j), imb, dw)
    end

    data1 = map(ts) do t
        imb = real(mapreduce(+, obs1) do ((i,j), o, d)
            return exp(-1im*t*(vals[j]-vals[i])) * o
        end)

        dw = real(mapreduce(+, obs1) do ((i,j), o, d)
            return exp(-1im*t*(vals[j]-vals[i])) * d
        end)

        return [t, imb, dw]
    end

    df1 = DataFrame(t = [d[1] for d in data1], imb = [d[2] for d in data1], dw = [d[3] for d in data1])
    CSV.write("../data/timeEv_8states_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df1)

    ### only 8 states ###
    println("2")
    is = collect(99:522)
    obs2 = map(Iterators.product(is,is)) do (i,j)
        imb = conj(vecs[init_idx,i])*vecs[init_idx,j]*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * imb_precalc[num]
        end
        dw = conj(vecs[init_idx,i])*vecs[init_idx,j]*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
            return conj(v1) * v2 * dw_precalc[num]
        end
        return ((i,j), imb, dw)
    end

    data2 = map(ts) do t
        imb = real(mapreduce(+, obs2) do ((i,j), o, d)
            return exp(-1im*t*(vals[j]-vals[i])) * o
        end)

        dw = real(mapreduce(+, obs2) do ((i,j), o, d)
            return exp(-1im*t*(vals[j]-vals[i])) * d
        end)

        return [t, imb, dw]
    end

    df2 = DataFrame(t = [d[1] for d in data2], imb = [d[2] for d in data2], dw = [d[3] for d in data2])
    CSV.write("../data/timeEv_dw8states_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df2)

    #continue

    #is2 = [en[1] for en in filter(d -> abs(d[2]-8) < 1e-1, collect(enumerate(dw)))]

    #@show is2
    #@show map(i -> dw[i], is2)

    occ = abs2.(vecs[476,:])
    #occRed = [(i,round(o,digits=4)) for (i,o) in enumerate(occ) if o > 1e-3]
    occRed = enumerate(occ)

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
