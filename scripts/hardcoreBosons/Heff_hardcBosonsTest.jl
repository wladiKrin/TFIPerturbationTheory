using CUDA, Distributed

addprocs(8)

gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0]

@everywhere begin 
    include("../../src/PertTheory.jl")

    include("./hardcoreBosons.jl")
    using CUDA, Distributed

    function singlerun(g)    
        #parameters
        J = -1;
        h = -0.;
        L= (8,8)
        N = prod(L);

        n = L[1]+1
        dt = 0.1
        T = 2e2

        single_dw_basis = map(1:n) do l
            s = fill(0,n)
            s[l] = 1
            return s
        end

        #initial state
        dw_basis = vec(collect(Iterators.product(fill(single_dw_basis,L[2])...)));
        init_idx = findall(x->x==Tuple(fill(single_dw_basis[div(n+1,2)], L[2])), dw_basis)[1]

        psi=zeros(length(dw_basis))
        psi[init_idx]=1

        imb_precalc = map(dw_basis) do dw
            return imbalance(dw, L, n)
        end
        dw_precalc = map(dw_basis) do dw
            return domainWallL(dw, (L,J,g,h))
        end

        println("finished precalc")

        basis_table = Dict(
            map(enumerate(dw_basis)) do (i, spin)
                return (spin, i)
            end
        );

        H = build_matrix(dw_basis, basis_table, (L,J,g,h))
        # dwM = domainWallL(dw_basis, basis_table, (L,J,g,h))
        # imbM = sparse(diagm(imb_precalc))
        println("finished building matrix")

        cuH    = cu(Array(H))
        cupsi  = cu(Array(psi))
        #cudwM  = cu(Array(dwM))
        #cuimbM = cu(Array(imbM))

        # Compute time evolution
        data = Any[]
        t = 0
        while t < T
          @time begin
              println(t)
              psi = sparse(Array(cupsi))
              imb = mapreduce(+, zip(findnz(psi)...)) do (i,v)
                  return abs2(v)*imb_precalc[i]
              end
              dw = mapreduce(+, zip(findnz(psi)...)) do (i,v)
                  return abs2(v)*dw_precalc[i]
              end

              append!(data, [[t, imb, dw]])
              
              # Propagate state
              cupsi,_ = exponentiate(cuH, -1im*dt, cupsi)
              t += dt
          end
          df = DataFrame(t = [d[1] for d in data], imb = [d[2] for d in data], dw = [d[3] for d in data])
          CSV.write("../../data/obs_Eff_Bound_hardcoreBosonsDiffEnergy_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g).csv", df);
        end

      df = DataFrame(t = [d[1] for d in data], imb = [d[2] for d in data], dw = [d[3] for d in data])
      CSV.write("../../data/obs_Eff_Bound_hardcoreBosonsDiffEnergy_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g).csv", df);
    end
end

# assign devices
asyncmap(zip(gs, workers(), devices())) do (g, w, d)
    remotecall_wait(w) do
        println("Worker $w uses , Parameters g = $(g)")
        device!(d)
        singlerun(g) 
        println("Finished")
    end
end
