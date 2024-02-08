using CUDA, Distributed

addprocs(8)

gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0]

@everywhere begin 
    # include("../../src/PertTheory.jl")

    include("./hardcoreBosons.jl")
    using CUDA.CUSPARSE

    function singlerun(g)    
        #parameters
        J = -1;
        h = -0.;
        L= (6,6)
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
        psi = sparse(psi)

        imb_precalc = map(dw_basis) do dw
            return imbalance(dw, L, n)
        end
        imbS1_precalc = map(dw_basis) do dw
            return imbalanceStrip(dw, L, n, 1)
        end
        imbS2_precalc = map(dw_basis) do dw
            return imbalanceStrip(dw, L, n, 2)
        end
        imbS3_precalc = map(dw_basis) do dw
            return imbalanceStrip(dw, L, n, 3)
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

        cuH    = CuSparseMatrixCSC(H)
        cupsi  = CuSparseVector(psi)
        cuDw  = CuSparseMatrixCSC(sparse(diagm(dw_precalc)))
        cuImb = CuSparseMatrixCSC(sparse(diagm(imb_precalc)))
        cuImb1 = CuSparseMatrixCSC(sparse(diagm(imbS1_precalc)))
        cuImb2 = CuSparseMatrixCSC(sparse(diagm(imbS2_precalc)))
        cuImb3 = CuSparseMatrixCSC(sparse(diagm(imbS3_precalc)))
        println("finished building cumatrix")

        # Compute time evolution
        data = Any[]
        t = 0
        while t < T
          @time begin
              println("time: ", t)
              psi = sparse(Array(cupsi))

              @time imb = dot(cupsi, cuImb * cupsi)
              @time imbS1 = dot(cupsi, cuImb1 * cupsi)
              @time imbS2 = dot(cupsi, cuImb2 * cupsi)
              @time imbS3 = dot(cupsi, cuImb3 * cupsi)
              @time dw = dot(cupsi, cuDw * cupsi)

              #=
              @time imbS1 = mapreduce(+, zip(findnz(psi)...)) do (i,v)
                return abs2(v) * imbS1_precalc[i]
              end
              println("finished imb1")

              @time imbS2 = mapreduce(+, zip(findnz(psi)...)) do (i,v)
                return abs2(v) * imbS2_precalc[i]
              end
              println("finished imb2")

              @time imbS3 = mapreduce(+, zip(findnz(psi)...)) do (i,v)
                return abs2(v) * imbS3_precalc[i]
              end
              println("finished imb3")

              @time dw = mapreduce(+, zip(findnz(psi)...)) do (i,v)
                return abs2(v) * dw_precalc[i]
              end
              println("finished dw")
              =#

              push!(data, [t, imb, imbS1, imbS2, imbS3, dw])

              # I, V = findnz(psi)
              # push!(data, [t, I, V])
              
              # Propagate state
              @time cupsi,_ = exponentiate(cuH, -1im*dt, cupsi)
              t += dt
          end
          name = "../../data/obs_Eff_Bound_hardcoreBosons_Strip_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g)"
          df = DataFrame(
            t = [real(d[1]) for d in data], 
            imb = [real(d[2]) for d in data], 
            imbS1 = [real(d[3]) for d in data], 
            imbS2 = [real(d[4]) for d in data], 
            imbS3 = [real(d[5]) for d in data], 
            dw = [real(d[6]) for d in data], 
          )
          h5open(name*".h5", "w") do file    
              file["time"] = df[:,"t"]
              file["imb"] = df[:,"imb"]
              file["imbS1"] = df[:,"imbS1"]
              file["imbS2"] = df[:,"imbS2"]
              file["imbS3"] = df[:,"imbS3"]
              file["dw"] = df[:,"dw"]
          end
          # CSV.write(name*".csv", df)
      end

      #df = DataFrame(t = [d[1] for d in data], imb = [d[2] for d in data], dw = [d[3] for d in data])
      #CSV.write("../../data/obs_Eff_Bound_hardcoreBosonsDiffEnergy_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g).csv", df);
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
