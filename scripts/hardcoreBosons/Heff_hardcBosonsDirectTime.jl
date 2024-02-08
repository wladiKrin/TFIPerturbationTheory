include("../../src/PertTheory.jl")
include("./hardcoreBosons.jl")

gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.]

let
    #parameters
    J = -1;
    # g = -1.;
    g = gs[parse(Int, ARGS[1])]
    h = -0.;
    L = (8,8)
    N = prod(L);

    n = 7
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

    basis_table = Dict(
        map(enumerate(dw_basis)) do (i, spin)
            return (spin, i)
        end
    );

	H = build_matrix(dw_basis, basis_table, (L,J,g,h))

	# Compute time evolution
	data = Any[]
	t = 0
	while t < T
      @time begin
          println("time: ", t)

          @time imb = mapreduce(+, zip(findnz(psi)...)) do (i,v)
            return abs2(v) * imb_precalc[i]
          end

          @time imbS1 = mapreduce(+, zip(findnz(psi)...)) do (i,v)
            return abs2(v) * imbS1_precalc[i]
          end

          @time imbS2 = mapreduce(+, zip(findnz(psi)...)) do (i,v)
            return abs2(v) * imbS2_precalc[i]
          end

          @time imbS3 = mapreduce(+, zip(findnz(psi)...)) do (i,v)
             return abs2(v) * imbS3_precalc[i]
          end

          @time dw = mapreduce(+, zip(findnz(psi)...)) do (i,v)
            return abs2(v) * dw_precalc[i]
          end

          push!(data, [t, imb, imbS1, imbS2, imbS3, dw])

          # I, V = findnz(psi)
          # push!(data, [t, I, V])
          
          # Propagate state
          @time psi,_ = exponentiate(H, -1im*dt, psi)
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
      CSV.write(name*".csv", df)
      # h5open(name*".h5", "w") do file    
      #     file["time"] = [real(d[1]) for d in data]
      #     file["inds"] = hcat([d[2] for d in data]...)
      #     file["vals"] = hcat([d[3] for d in data]...)
      # end
	end

  # name = "../../data/obs_Eff_Bound_hardcoreBosons_Strip_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g)"
  # df = DataFrame(
  #   t = [real(d[1]) for d in data], 
  #   imb = [real(d[2]) for d in data], 
  #   imbS1 = [real(d[3]) for d in data], 
  #   imbS2 = [real(d[4]) for d in data], 
  #   dw = [real(d[5]) for d in data], 
  # )
  # CSV.write(name*".csv", df)

  #= 
  h5open(name*".h5", "w") do file    
      file["time"] = [real(d[1]) for d in data]
      file["psi"] = hcat([d[2] for d in data]...)
  end
  =# 
end
