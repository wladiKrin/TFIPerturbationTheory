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

    n = 5
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
        return imbalanceStrip(dw, L, n)
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
            println(t)

            append!(data, [[t, psi]])
            
            # Propagate state
            psi,_ = exponentiate(H, -1im*dt, psi)
            t += dt
        end
	end

  name = "../../data/obs_Eff_Bound_hardcoreBosons_Strip_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g)"
  h5open(name*".h5", "w") do file    
      file["time"] = [real(d[1]) for d in data]
      file["psi"] = hcat([d[2] for di in data]...)
  end
end
